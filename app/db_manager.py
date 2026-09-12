"""
=====================================================================
 MODULE DB_MANAGER — AUTHENTIFICATION & PERSISTANCE SUPABASE
=====================================================================
"""

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import hashlib
import secrets
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


def create_password_reset_token(email: str) -> Optional[str]:
    """Crée un jeton de réinitialisation valable une heure pour un email connu."""
    email = (email or "").strip().lower()
    if not email:
        return None

    result = execute_query(
        "SELECT id FROM public.users WHERE LOWER(email) = LOWER(%s) AND actif = TRUE LIMIT 1",
        (email,), fetch=True,
    )
    if not result:
        return None

    user_id = dict(result[0])["id"]
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    execute_query(
        "DELETE FROM public.password_reset_tokens WHERE user_id = %s OR date_expiration <= CURRENT_TIMESTAMP",
        (user_id,),
    )
    created = execute_query(
        """
        INSERT INTO public.password_reset_tokens
            (user_id, token_hash, date_expiration)
        VALUES (%s, %s, CURRENT_TIMESTAMP + INTERVAL '1 hour')
        RETURNING id
        """,
        (user_id, token_hash), fetch=True,
    )
    return raw_token if created else None


def reset_password_with_token(token: str, new_password: str) -> bool:
    """Remplace le mot de passe et invalide le jeton utilisé."""
    token = (token or "").strip()
    if len(new_password or "") < 8 or not token:
        return False

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    password_hash = hash_password(new_password)
    result = execute_query(
        """
        UPDATE public.users AS u
        SET password_hash = %s
        FROM public.password_reset_tokens AS r
        WHERE r.user_id = u.id
          AND r.token_hash = %s
          AND r.date_utilisation IS NULL
          AND r.date_expiration > CURRENT_TIMESTAMP
        RETURNING u.id
        """,
        (password_hash, token_hash), fetch=True,
    )
    if not result:
        return False

    execute_query(
        "UPDATE public.password_reset_tokens SET date_utilisation = CURRENT_TIMESTAMP WHERE token_hash = %s",
        (token_hash,),
    )
    return True


def login_user(identifiant: str, password: str, remember_me: bool = False) -> Optional[Dict]:
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
    duree_session = 24 * 60 * 60 if remember_me else 8 * 60 * 60
    session_query = """
        INSERT INTO public.sessions
            (user_id, token_session, date_connexion, duree_session_secondes, actif)
        VALUES (%s, %s, CURRENT_TIMESTAMP, %s, TRUE)
    """
    execute_query(session_query, (user["id"], session_id, duree_session))

    user["session_id"] = session_id
    return user


def get_user_by_session(session_id: str) -> Optional[Dict]:
    """Restaure un utilisateur depuis une session persistante encore valide."""
    session_id = (session_id or "").strip()
    if not session_id:
        return None

    query = """
        SELECT
            u.id, u.email, u.nom_complet, u.institution, u.role,
            u.actif, u.date_derniere_connexion, u.nb_connexions_total,
            s.token_session
        FROM public.sessions AS s
        INNER JOIN public.users AS u ON u.id = s.user_id
        WHERE s.token_session = %s
          AND s.actif = TRUE
          AND u.actif = TRUE
          AND s.date_connexion + (s.duree_session_secondes * INTERVAL '1 second') > CURRENT_TIMESTAMP
        LIMIT 1
    """
    result = execute_query(query, (session_id,), fetch=True)
    if not result:
        return None

    user = dict(result[0])
    user["session_id"] = user["token_session"]
    return user


def register_user(email: str, password: str, nom: str, prenom: str,
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
    
    nom = (nom or "").strip()
    prenom = (prenom or "").strip()
    nom_complet = f"{nom} {prenom}".strip()

    # Insérer l'utilisateur
    insert_query = """
        INSERT INTO public.users 
        (email, password_hash, nom, prenom, nom_complet, institution, role, actif, date_creation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
        RETURNING id
    """
    
    result = execute_query(
        insert_query,
        (email, password_hash, nom, prenom, nom_complet, institution, role),
        fetch=True,
    )
    
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
        'duree_mois', 'objet_pret', 'objet_pret_justification',
        'secteur_activite', 'secteur_activite_justification', 'anciennete_activite',
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
        int(data.get('duree', 0)), data.get('objet'), data.get('objet_justification'),
        data.get('secteur'), data.get('secteur_justification'), int(data.get('anciennete', 0)),
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
    archivees: bool = False,
) -> List[Dict]:
    """
    Récupère les demandes de crédit selon le rôle et l'institution de l'utilisateur.

    LOGIQUE D'ACCÈS UNIFIÉE :
    ✅ Admin : voit TOUTES les demandes de toutes les institutions
    ✅ Manager : voit toutes les demandes de son institution
    ✅ Agent : voit TOUTES les demandes de son institution (PAS juste les siennes)

    ⚠️ IMPORTANT MÉTIER : Tous les agents d'une même institution voient le MÊME
    historique complet. Cela facilite la collaboration et la traçabilité.

    `archivees` : si False (par défaut), exclut les demandes mises à la
    corbeille (statut='archivee'). Si True, ne renvoie QUE celles-ci
    (vue "Corbeille"). Le statut 'archivee' sert de suppression douce :
    la ligne reste en base, juste masquée de l'historique normal.
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
    elif archivees:
        conditions.append("d.statut = 'archivee'")
    else:
        conditions.append("COALESCE(d.statut, '') != 'archivee'")

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
    """Récupère tous les détails d'une demande, à partir de son id_demande
    (l'identifiant affiché partout dans l'interface, ex. "#20260911-0001"),
    pas de la clé primaire interne."""
    query = "SELECT * FROM public.demandes_credit WHERE id_demande = %s LIMIT 1"
    result = execute_query(query, (demande_id,), fetch=True)
    return dict(result[0]) if result and len(result) > 0 else None


def archiver_demande(id_demande: str) -> bool:
    """Met une demande à la corbeille (suppression douce) : passe son
    statut à 'archivee'. La ligne n'est jamais effacée de la base, elle
    est seulement exclue de get_demandes() par défaut."""
    query = "UPDATE public.demandes_credit SET statut = 'archivee' WHERE id_demande = %s"
    execute_query(query, (id_demande,))
    return True


def restaurer_demande(id_demande: str) -> bool:
    """Sort une demande de la corbeille : remet son statut à 'analysee'."""
    query = "UPDATE public.demandes_credit SET statut = 'analysee' WHERE id_demande = %s"
    execute_query(query, (id_demande,))
    return True


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