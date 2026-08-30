"""
=====================================================================
 MODULE DB_MANAGER — AUTHENTIFICATION & PERSISTANCE SUPABASE
=====================================================================
Gère toute interaction avec Supabase (PostgreSQL hébergé).
À importer dans l'app Streamlit : from db_manager import *

Fonctions principales :
  • Authentication : login_user(), register_user(), logout_user()
  • Demandes : save_demande(), get_demandes(), get_demande_detail()
  • Agents : get_agent_info(), update_agent_stats()
  • Utilitaires : get_user_full_info()
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


def login_user(identifiant: str, password: str) -> Optional[Dict]:
    """
    Authentifie un utilisateur par email ou nom complet.

    Le mot de passe est vérifié avec bcrypt et une session applicative
    est créée en cas de succès.
    """
    identifiant = (identifiant or "").strip()
    if not identifiant or not password:
        return None

    query = """
        SELECT
            u.id,
            u.email,
            u.password_hash,
            u.nom_complet,
            u.institution,
            u.role,
            u.actif,
            u.date_derniere_connexion,
            u.nb_connexions_total
        FROM public.users AS u
        WHERE (LOWER(u.email) = LOWER(%s)
               OR LOWER(u.nom_complet) = LOWER(%s))
          AND u.actif = TRUE
        LIMIT 1
    """

    result = execute_query(query, (identifiant, identifiant), fetch=True)
    if not result:
        return None

    user = dict(result[0])
    password_hash = user.pop("password_hash", "") or ""

    if not verify_password(password, password_hash):
        return None

    update_query = """
        UPDATE public.users
        SET date_derniere_connexion = CURRENT_TIMESTAMP,
            nb_connexions_total = COALESCE(nb_connexions_total, 0) + 1
        WHERE id = %s
    """
    execute_query(update_query, (user["id"],))

    session_id = str(uuid.uuid4())
    session_query = """
        INSERT INTO public.sessions
            (user_id, token_session, date_connexion, actif)
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
            code_agent = f"AGT-{str(user_id).replace('-', '')[:8].upper()}"
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
def get_demandes(
    user_id: Optional[str] = None,
    statut: str = None,
    limit: int = 50,
    role: str = "agent",
    institution: str = None,
) -> List[Dict]:
    """
    Récupère les demandes de crédit selon le rôle et l'institution de l'utilisateur.
    
    LOGIQUE D'ACCÈS UNIFIÉE :
    ✅ Admin : voit TOUTES les demandes de toutes les institutions
    ✅ Manager : voit toutes les demandes de son institution
    ✅ Agent : voit TOUTES les demandes de son institution (PAS juste les siennes)
    
    ⚠️ IMPORTANT MÉTIER : Tous les agents d'une même institution voient le MÊME
    historique complet. Cela facilite la collaboration et la traçabilité.
    """
    conditions = []
    params = []

    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        limit = 50

    role = (role or "agent").lower()

    # Les admins voient tout, les autres sont filtrés par institution
    if role != "admin":
        if not institution:
            return []
        # ✅ Tous les agents et managers de l'institution voient toutes les demandes
        conditions.append("u.institution = %s")
        params.append(institution)

    if statut:
        conditions.append("d.statut = %s")
        params.append(statut)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
        SELECT
            d.id,
            d.id_demande,
            d.user_id,
            d.nom_demandeur,
            d.prenom_demandeur,
            d.age_tranche,
            d.secteur_activite,
            d.montant_demande,
            d.montant_accorde,
            d.score_ml,
            d.categorie_risque,
            d.decision,
            d.statut,
            d.date_creation,
            d.date_analyse
        FROM public.demandes_credit AS d
        INNER JOIN public.users AS u
            ON u.id = d.user_id
        {where_clause}
        ORDER BY d.date_creation DESC
        LIMIT %s
    """

    params.append(limit)
    return execute_query(query, tuple(params), fetch=True) or []


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
@st.cache_data(ttl=10, show_spinner=False)
def get_demande_historique(demande_id: str) -> List[Dict]:
    """Récupère l'historique complet d'une demande."""
    query = """
        SELECT * FROM public.demandes_historique
        WHERE demande_id = %s
        ORDER BY date_action DESC
    """
    return execute_query(query, (demande_id,), fetch=True) or []


# =====================================================================
# 5. UTILITAIRES
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