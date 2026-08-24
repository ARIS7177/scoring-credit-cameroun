"""
=====================================================================
 MODULE DB_MANAGER — AUTHENTIFICATION & PERSISTANCE SUPABASE
=====================================================================
Gère toute interaction avec Supabase (PostgreSQL hébergé).
À importer dans l'app Streamlit : from db_manager import *

Fonctions principales :
  • Authentication : login_user(), register_user(), logout_user()
  • Demandes : save_demande(), get_demandes(), update_demande_status()
  • Agents : get_agent_info(), update_agent_stats()
  • Historique : log_action()
=====================================================================
"""

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import uuid
from functools import wraps

# =====================================================================
# 0. CONFIGURATION SUPABASE
# =====================================================================

def get_supabase_connection():
    """
    Établit une connexion PostgreSQL à Supabase.
    Les identifiants viennent de st.secrets (fichier .streamlit/secrets.toml)
    """
    try:
        # Récupérer l'URL Supabase et extraire les paramètres de connexion
        supabase_config = st.secrets.get("supabase", {})
        supabase_url = supabase_config.get("url")
        
        if not supabase_url:
            raise ValueError("⚠️ Supabase URL non configurée dans st.secrets")
        
        pooler_host = supabase_config.get("pooler_host")
        db_user = supabase_config.get("db_user") or "postgres"
        db_port = int(supabase_config.get("db_port", 5432))
        db_name = supabase_config.get("db_name") or "postgres"
        if not pooler_host:
            raise ValueError("pooler_host non configuré dans st.secrets")
        
        # Le password Supabase est dans service_role_key ou un password séparé
        db_password = supabase_config.get("db_password")
        if not db_password:
            raise ValueError("db_password non configuré dans st.secrets")
        
        # Établir la connexion
        conn = psycopg2.connect(
            host=pooler_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=5,
            sslmode="require"
        )
        return conn
    except Exception as e:
        st.error(f"❌ Erreur connexion Supabase : {e}")
        return None


def execute_query(query: str, params: Tuple = None, fetch: bool = False):
    """
    Exécute une requête SQL sur Supabase.
    
    Args:
        query: Requête SQL
        params: Paramètres liés (tuple)
        fetch: Si True, retourne les résultats
    
    Returns:
        Résultats si fetch=True, None sinon
    """
    conn = get_supabase_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
            conn.commit()
        return result
    except Exception as e:
        st.error(f"❌ Erreur requête DB : {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


# =====================================================================
# 1. AUTHENTIFICATION
# =====================================================================

def hash_password(password: str) -> str:
    """Hache un mot de passe avec bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Vérifie un mot de passe contre son hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def login_user(email: str, password: str) -> Optional[Dict]:
    """
    Authentifie un utilisateur contre la base Supabase.
    
    Returns:
        Dict avec infos user si succès, None sinon
    """
    query = """
        SELECT id, email, nom_complet, institution, role, actif, 
               date_derniere_connexion, nb_connexions_total
        FROM public.users
        WHERE email = %s AND actif = TRUE
        LIMIT 1
    """
    
    result = execute_query(query, (email,), fetch=True)
    if not result or len(result) == 0:
        return None
    
    user = dict(result[0])
    
    # Récupérer le hash du password (besoin d'une requête séparée pour la sécurité)
    hash_query = "SELECT password_hash FROM public.users WHERE email = %s"
    hash_result = execute_query(hash_query, (email,), fetch=True)
    
    if not hash_result:
        return None
    
    password_hash = dict(hash_result[0])["password_hash"]
    
    # Vérifier le mot de passe
    if not verify_password(password, password_hash):
        return None
    
    # Mettre à jour last_login
    update_query = """
        UPDATE public.users 
        SET date_derniere_connexion = CURRENT_TIMESTAMP,
            nb_connexions_total = nb_connexions_total + 1
        WHERE email = %s
    """
    execute_query(update_query, (email,))
    
    # Créer une session
    session_id = str(uuid.uuid4())
    session_query = """
        INSERT INTO public.sessions (user_id, token_session, date_connexion, actif)
        VALUES (%s, %s, CURRENT_TIMESTAMP, TRUE)
    """
    execute_query(session_query, (user["id"], session_id))
    
    user["session_id"] = session_id
    return user


def register_user(email: str, password: str, nom_complet: str, 
                  institution: str, role: str = "agent") -> Tuple[bool, str]:
    """
    Crée un nouvel utilisateur dans Supabase.
    
    Returns:
        (succès: bool, message: str)
    """
    # Vérifier que l'email n'existe pas
    check_query = "SELECT id FROM public.users WHERE email = %s"
    result = execute_query(check_query, (email,), fetch=True)
    if result and len(result) > 0:
        return False, "Cet email est déjà enregistré"
    
    # Hasher le password
    password_hash = hash_password(password)
    
    # Insérer l'utilisateur
    insert_query = """
        INSERT INTO public.users 
        (email, password_hash, nom_complet, institution, role, actif, date_creation)
        VALUES (%s, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
        RETURNING id
    """
    
    result = execute_query(insert_query, (email, password_hash, nom_complet, institution, role), fetch=True)
    
    if result and len(result) > 0:
        user_id = dict(result[0])["id"]
        
        # Si agent, créer l'enregistrement agent
        if role == "agent":
            agent_query = """
                INSERT INTO public.agents (id, user_id, code_agent)
                VALUES (%s, %s, %s)
            """
            code_agent = f"AGT-{user_id.hex[:8].upper()}"
            execute_query(agent_query, (user_id, user_id, code_agent))
        
        return True, "Utilisateur créé avec succès"
    else:
        return False, "Erreur lors de la création de l'utilisateur"


def logout_user(user_id: str, session_id: str):
    """Ferme la session utilisateur."""
    update_query = """
        UPDATE public.sessions 
        SET date_deconnexion = CURRENT_TIMESTAMP, actif = FALSE
        WHERE user_id = %s AND token_session = %s
    """
    execute_query(update_query, (user_id, session_id))


# =====================================================================
# 2. GESTION DES DEMANDES DE CRÉDIT
# =====================================================================

def save_demande(data: Dict, user_id: str) -> Optional[str]:
    """
    Enregistre une nouvelle demande de crédit dans Supabase.
    
    Args:
        data: Dict avec tous les champs de la demande
        user_id: ID de l'agent qui crée la demande
    
    Returns:
        ID demande si succès, None sinon
    """
    # Générer l'ID demande
    id_demande = f"#{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    
    # Préparer les colonnes
    columns = [
        'id_demande', 'user_id', 'nom_demandeur', 'prenom_demandeur',
        'genre', 'age_tranche', 'education', 'adresse_demandeur',
        'revenu_mensuel', 'charges_mensuelles', 'montant_demande',
        'duree_mois', 'objet_pret', 'secteur_activite', 'anciennete_activite',
        'score_ml', 'categorie_risque', 'proba_defaut', 'decision', 'source_score',
        'activite_saisonniere', 'mobile_money', 'ligne_credit_ouverte', 'usage_credit',
        'garant', 'logement_situation', 'statut', 'date_creation', 'date_analyse'
    ]
    
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    
    query = f"""
        INSERT INTO public.demandes_credit ({columns_str})
        VALUES ({placeholders})
        RETURNING id, id_demande
    """
    
    # Préparer les valeurs
    values = [
        id_demande, user_id,
        data.get('nom'), data.get('prenom'),
        data.get('genre'), data.get('age'), data.get('education'), data.get('adresse'),
        float(data.get('revenu', 0)), float(data.get('charges', 0)), float(data.get('montant_demande', 0)),
        int(data.get('duree', 0)), data.get('objet'), data.get('secteur'), int(data.get('anciennete', 0)),
        int(data.get('score_ml', 0)) if data.get('score_ml') else None,
        data.get('categorie_risque'), float(data.get('proba_defaut', 0)) if data.get('proba_defaut') else None,
        data.get('decision'), data.get('source_score', 'HEURISTIQUE'),
        data.get('activite_saisonniere') == 'Oui',
        data.get('mobile_money') == 'Oui',
        data.get('ligne_credit') == 'Oui',
        data.get('usage_credit'),
        data.get('garant'), data.get('logement'),
        'analysee' if data.get('score_ml') else 'saisie',
        datetime.now(),
        datetime.now() if data.get('score_ml') else None
    ]
    
    result = execute_query(query, tuple(values), fetch=True)
    
    if result and len(result) > 0:
        record = dict(result[0])
        return record['id_demande']
    return None


@st.cache_data(ttl=10, show_spinner=False)
def get_demandes(user_id: str, statut: str = None, limit: int = 50) -> List[Dict]:
    """
    Récupère les demandes d'un agent.
    
    Args:
        user_id: ID de l'agent
        statut: Filtrer par statut (optionnel)
        limit: Nombre maximal de résultats
    
    Returns:
        Liste de demandes
    """
    where_clause = "WHERE user_id = %s"
    params = [user_id]
    
    if statut:
        where_clause += " AND statut = %s"
        params.append(statut)
    
    query = f"""
         SELECT id, id_demande, nom_demandeur, prenom_demandeur, age_tranche,
             secteur_activite, montant_demande, montant_accorde, score_ml,
             categorie_risque, decision, statut,
               date_creation, date_analyse
        FROM public.demandes_credit
        {where_clause}
        ORDER BY date_creation DESC
        LIMIT %s
    """
    
    params.append(limit)
    return execute_query(query, tuple(params), fetch=True) or []


def update_demande_status(demande_id: str, new_status: str, 
                         user_id: str, notes: str = None) -> bool:
    """
    Met à jour le statut d'une demande et enregistre l'historique.
    """
    # Mettre à jour la demande
    update_query = """
        UPDATE public.demandes_credit
        SET statut = %s, date_decision = CURRENT_TIMESTAMP, notes_agent = %s
        WHERE id = %s
    """
    
    result = execute_query(update_query, (new_status, notes, demande_id))
    
    if result is not None:
        # Enregistrer dans l'historique
        log_action(demande_id, user_id, 'decision', f'Statut changé à {new_status}', notes)
        return True
    return False


def get_demande_detail(demande_id: str) -> Optional[Dict]:
    """Récupère tous les détails d'une demande."""
    query = "SELECT * FROM public.demandes_credit WHERE id = %s LIMIT 1"
    result = execute_query(query, (demande_id,), fetch=True)
    return dict(result[0]) if result and len(result) > 0 else None


# =====================================================================
# 3. GESTION DES AGENTS
# =====================================================================

def get_agent_info(user_id: str) -> Optional[Dict]:
    """Récupère les infos détaillées d'un agent."""
    query = """
        SELECT 
            a.id, a.user_id, a.code_agent, a.competences,
            a.demandes_traitees_total, a.demandes_acceptees, a.demandes_refusees,
            a.score_performance,
            u.nom_complet, u.email, u.institution, u.date_derniere_connexion
        FROM public.agents a
        LEFT JOIN public.users u ON a.user_id = u.id
        WHERE a.user_id = %s
        LIMIT 1
    """
    result = execute_query(query, (user_id,), fetch=True)
    return dict(result[0]) if result and len(result) > 0 else None


def update_agent_stats(user_id: str, decision: str):
    """Mets à jour les stats d'un agent après une décision."""
    query = """
        UPDATE public.agents
        SET demandes_traitees_total = demandes_traitees_total + 1
    """
    
    if decision == "ACCORDÉ":
        query += ", demandes_acceptees = demandes_acceptees + 1"
    elif decision == "REFUSÉ":
        query += ", demandes_refusees = demandes_refusees + 1"
    
    query += " WHERE user_id = %s"
    
    execute_query(query, (user_id,))


# =====================================================================
# 4. HISTORIQUE ET AUDIT TRAIL
# =====================================================================

def log_action(demande_id: str, user_id: str, action: str, 
               notes: str = None, statut_nouveau: str = None):
    """
    Enregistre une action dans l'historique (audit trail).
    """
    query = """
        INSERT INTO public.demandes_historique 
        (demande_id, user_id, action, notes, statut_nouveau, date_action)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """
    
    execute_query(query, (demande_id, user_id, action, notes, statut_nouveau))


def get_demande_historique(demande_id: str) -> List[Dict]:
    """Récupère l'historique complet d'une demande."""
    query = """
        SELECT * FROM public.demandes_historique
        WHERE demande_id = %s
        ORDER BY date_action DESC
    """
    return execute_query(query, (demande_id,), fetch=True) or []


# =====================================================================
# 5. STATISTIQUES QUOTIDIENNES
# =====================================================================

def get_stats_institution(institution: str, date: str = None) -> Optional[Dict]:
    """
    Récupère les stats quotidiennes d'une institution.
    
    Args:
        institution: Nom de l'institution
        date: Date au format YYYY-MM-DD (défaut: aujourd'hui)
    
    Returns:
        Dict avec les stats
    """
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    query = """
        SELECT * FROM public.stats_quotidiennes
        WHERE institution = %s AND date_stat = %s
        LIMIT 1
    """
    
    result = execute_query(query, (institution, date), fetch=True)
    return dict(result[0]) if result and len(result) > 0 else None


def update_stats_quotidiennes(institution: str, user_id: str = None):
    """
    Recalcule les stats quotidiennes pour une institution.
    À appeler après chaque décision de crédit.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Récupérer les stats depuis les demandes du jour
    stats_query = """
        SELECT
            COUNT(*) as nb_creees,
            COUNT(CASE WHEN score_ml IS NOT NULL THEN 1 END) as nb_analysees,
            COUNT(CASE WHEN decision = 'ACCORDÉ' THEN 1 END) as nb_acceptees,
            COUNT(CASE WHEN decision = 'REFUSÉ' THEN 1 END) as nb_refusees,
            COUNT(CASE WHEN decision = 'ÉTUDE APPROFONDIE' THEN 1 END) as nb_etude,
            COALESCE(SUM(montant_demande), 0) as montant_total_demande,
            COALESCE(SUM(montant_accorde), 0) as montant_total_accorde,
            ROUND(AVG(score_ml), 2) as score_moyen
        FROM public.demandes_credit
        WHERE institution = %s AND DATE(date_creation) = %s
    """
    
    stats_result = execute_query(stats_query, (institution, today), fetch=True)
    
    if stats_result and len(stats_result) > 0:
        stats = dict(stats_result[0])
        
        # Calculer taux acceptation
        nb_total = stats['nb_analysees']
        taux_acceptation = (stats['nb_acceptees'] / nb_total * 100) if nb_total > 0 else 0
        
        # Upsert les stats
        upsert_query = """
            INSERT INTO public.stats_quotidiennes
            (institution, user_id, date_stat, nb_demandes_creees, nb_demandes_analysees,
             nb_demandes_acceptees, nb_demandes_refusees, nb_demandes_etude,
             montant_total_demande, montant_total_accorde, score_moyen, taux_acceptation,
             date_maj)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (institution, date_stat, user_id) DO UPDATE SET
                nb_demandes_creees = EXCLUDED.nb_demandes_creees,
                nb_demandes_analysees = EXCLUDED.nb_demandes_analysees,
                nb_demandes_acceptees = EXCLUDED.nb_demandes_acceptees,
                nb_demandes_refusees = EXCLUDED.nb_demandes_refusees,
                nb_demandes_etude = EXCLUDED.nb_demandes_etude,
                montant_total_demande = EXCLUDED.montant_total_demande,
                montant_total_accorde = EXCLUDED.montant_total_accorde,
                score_moyen = EXCLUDED.score_moyen,
                taux_acceptation = EXCLUDED.taux_acceptation,
                date_maj = CURRENT_TIMESTAMP
        """
        
        execute_query(upsert_query, (
            institution, user_id, today,
            stats['nb_creees'], stats['nb_analysees'],
            stats['nb_acceptees'], stats['nb_refusees'], stats['nb_etude'],
            stats['montant_total_demande'], stats['montant_total_accorde'],
            stats['score_moyen'], taux_acceptation
        ))


# =====================================================================
# 6. UTILITAIRES
# =====================================================================

def require_auth(f):
    """Décorateur pour protéger une fonction si l'utilisateur n'est pas authentifié."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated"):
            st.error("❌ Vous devez être connecté")
            return None
        return f(*args, **kwargs)
    return wrapper


def get_user_full_info(user_id: str) -> Optional[Dict]:
    """Récupère les infos complètes d'un utilisateur + agent."""
    query = """
        SELECT 
            u.id, u.email, u.nom_complet, u.institution, u.role, u.actif,
            u.date_derniere_connexion, u.nb_connexions_total,
            a.code_agent, a.competences, a.demandes_traitees_total,
            a.demandes_acceptees, a.demandes_refusees, a.score_performance
        FROM public.users u
        LEFT JOIN public.agents a ON u.id = a.user_id
        WHERE u.id = %s
        LIMIT 1
    """
    
    result = execute_query(query, (user_id,), fetch=True)
    return dict(result[0]) if result and len(result) > 0 else None


def health_check() -> bool:
    """Vérifie que la connexion à Supabase fonctionne."""
    query = "SELECT 1"
    result = execute_query(query, fetch=True)
    return result is not None


