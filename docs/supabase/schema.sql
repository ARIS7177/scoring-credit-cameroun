-- SQLBook: Code
-- =====================================================================
-- SCHÉMA SUPABASE — SYSTÈME DE SCORING CRÉDIT CAMEROUN
-- =====================================================================
-- Base de données PostgreSQL pour authentification réelle + persistance
-- À exécuter dans l'éditeur SQL de Supabase
-- =====================================================================

-- --- EXTENSION UUID ---
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================================
-- TABLE 1 : UTILISATEURS (agents de microfinance)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    nom VARCHAR(150),
    prenom VARCHAR(150),
    nom_complet VARCHAR(255) NOT NULL,
    institution VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'agent', -- 'agent', 'admin', 'manager'
    actif BOOLEAN DEFAULT TRUE,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_derniere_connexion TIMESTAMP,
    nb_connexions_total INTEGER DEFAULT 0,
    CONSTRAINT email_format CHECK (
        email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    )
);

CREATE INDEX idx_users_email ON public.users (email);

CREATE INDEX idx_users_institution ON public.users (institution);

CREATE INDEX idx_users_actif ON public.users (actif);

-- Migration pour les bases déjà créées avant la séparation nom/prénom.
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS nom VARCHAR(150);
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS prenom VARCHAR(150);

UPDATE public.users
SET nom = COALESCE(nom, split_part(nom_complet, ' ', 1)),
    prenom = COALESCE(
        prenom,
        NULLIF(TRIM(SUBSTRING(nom_complet FROM POSITION(' ' IN nom_complet) + 1)), '')
    )
WHERE nom IS NULL OR prenom IS NULL;

-- =====================================================================
-- TABLE 2 : AGENTS (copie dénormalisée pour performances)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.agents (
    id UUID PRIMARY KEY REFERENCES public.users (id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users (id) ON DELETE CASCADE,
    code_agent VARCHAR(50) UNIQUE,
    phone VARCHAR(20),
    adresse VARCHAR(500),
    date_embauche DATE,
    competences VARCHAR(500),
    demandes_traitees_total INTEGER DEFAULT 0,
    demandes_acceptees INTEGER DEFAULT 0,
    demandes_refusees INTEGER DEFAULT 0,
    score_performance DECIMAL(5, 2) DEFAULT 0.0,
    date_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agents_code ON public.agents (code_agent);

CREATE INDEX idx_agents_institution ON public.agents (user_id);

-- =====================================================================
-- TABLE 3 : DEMANDES DE CRÉDIT (principal)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.demandes_credit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    id_demande VARCHAR(50) UNIQUE NOT NULL, -- Format : #YYYYMMDD-0001
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES public.agents(id) ON DELETE SET NULL,
    institution VARCHAR(255),

-- Informations demandeur
nom_demandeur VARCHAR(255) NOT NULL,
prenom_demandeur VARCHAR(255) NOT NULL,
genre VARCHAR(20),
age_tranche VARCHAR(20),
education VARCHAR(50),
adresse_demandeur VARCHAR(500),
telephone VARCHAR(20),
email_demandeur VARCHAR(255),

-- Capacité financière
revenu_mensuel DECIMAL(15, 2),
charges_mensuelles DECIMAL(15, 2),
ratio_endettement DECIMAL(5, 2),

-- Demande de crédit
montant_demande DECIMAL(15, 2) NOT NULL,
montant_accorde DECIMAL(15, 2),
montant_recommande DECIMAL(15, 2),
duree_mois INTEGER,
objet_pret VARCHAR(100),
    objet_pret_justification TEXT,
taux_interesse DECIMAL(5, 2),

-- Détails professionnels
secteur_activite VARCHAR(100),
    secteur_activite_justification TEXT,
anciennete_activite INTEGER,
activite_saisonniere BOOLEAN DEFAULT FALSE,
mobile_money BOOLEAN DEFAULT FALSE,
ligne_credit_ouverte BOOLEAN DEFAULT FALSE,
usage_credit VARCHAR(50),

-- Garanties
garant VARCHAR(200), logement_situation VARCHAR(50),

-- Résultats ML
score_ml INTEGER, -- 0-100
categorie_risque VARCHAR(20), -- FAIBLE, MODÉRÉ, ÉLEVÉ
proba_defaut DECIMAL(5, 2),
decision VARCHAR(50), -- ACCORDÉ, ÉTUDE, REFUSÉ
source_score VARCHAR(20), -- ML ou HEURISTIQUE
facteurs_explicatifs TEXT, -- JSON array of factors

-- Métadonnées
statut VARCHAR(50) DEFAULT 'saisie', -- saisie, analysee, accordee, refusee, etude, archivee
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_analyse TIMESTAMP,
    date_decision TIMESTAMP,
    notes_agent TEXT,
    
    CONSTRAINT montant_positif CHECK (montant_demande > 0),
    CONSTRAINT duree_positive CHECK (duree_mois > 0)
);

CREATE INDEX idx_demandes_user ON public.demandes_credit (user_id);

CREATE INDEX idx_demandes_agent ON public.demandes_credit (agent_id);

CREATE INDEX idx_demandes_id_demande ON public.demandes_credit (id_demande);

CREATE INDEX idx_demandes_statut ON public.demandes_credit (statut);

CREATE INDEX idx_demandes_date_creation ON public.demandes_credit (date_creation DESC);

CREATE INDEX idx_demandes_score ON public.demandes_credit (score_ml);

CREATE INDEX idx_demandes_decision ON public.demandes_credit (decision);

CREATE INDEX idx_demandes_institution ON public.demandes_credit (institution);

-- Migration pour les bases déjà créées avant l'ajout des justifications.
ALTER TABLE public.demandes_credit
    ADD COLUMN IF NOT EXISTS objet_pret_justification TEXT;
ALTER TABLE public.demandes_credit
    ADD COLUMN IF NOT EXISTS secteur_activite_justification TEXT;

-- =====================================================================
-- TABLE 4 : HISTORIQUE DES DEMANDES (audit trail)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.demandes_historique (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    demande_id UUID NOT NULL REFERENCES public.demandes_credit (id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
    statut_ancien VARCHAR(50),
    statut_nouveau VARCHAR(50),
    montant_ancien DECIMAL(15, 2),
    montant_nouveau DECIMAL(15, 2),
    decision_ancien VARCHAR(50),
    decision_nouveau VARCHAR(50),
    action VARCHAR(100), -- 'creation', 'modification', 'decision', 'archivage'
    notes TEXT,
    ip_adresse VARCHAR(50),
    date_action TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_historique_demande ON public.demandes_historique (demande_id);

CREATE INDEX idx_historique_user ON public.demandes_historique (user_id);

CREATE INDEX idx_historique_date ON public.demandes_historique (date_action DESC);

-- =====================================================================
-- TABLE 5 : SESSIONS (suivi des connexions)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    user_id UUID NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
    token_session VARCHAR(500) UNIQUE,
    ip_adresse VARCHAR(50),
    user_agent TEXT,
    date_connexion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_deconnexion TIMESTAMP,
    duree_session_secondes INTEGER,
    actif BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_user ON public.sessions (user_id);

CREATE INDEX idx_sessions_date ON public.sessions (date_connexion DESC);

CREATE INDEX idx_sessions_actif ON public.sessions (actif);

-- =====================================================================
-- TABLE 6 : JETONS DE RÉINITIALISATION DU MOT DE PASSE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    user_id UUID NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_expiration TIMESTAMP NOT NULL,
    date_utilisation TIMESTAMP,
    CONSTRAINT token_expiration_future CHECK (
        date_expiration > date_creation
    )
);

CREATE INDEX idx_password_reset_tokens_user ON public.password_reset_tokens (user_id);

CREATE INDEX idx_password_reset_tokens_expiration ON public.password_reset_tokens (date_expiration);

-- =====================================================================
-- TABLE 6 : STATISTIQUES QUOTIDIENNES (cache pour dashboard)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.stats_quotidiennes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    institution VARCHAR(255),
    user_id UUID REFERENCES public.users (id),
    date_stat DATE DEFAULT CURRENT_DATE,
    nb_demandes_creees INTEGER DEFAULT 0,
    nb_demandes_analysees INTEGER DEFAULT 0,
    nb_demandes_acceptees INTEGER DEFAULT 0,
    nb_demandes_refusees INTEGER DEFAULT 0,
    nb_demandes_etude INTEGER DEFAULT 0,
    montant_total_demande DECIMAL(15, 2) DEFAULT 0,
    montant_total_accorde DECIMAL(15, 2) DEFAULT 0,
    score_moyen DECIMAL(5, 2) DEFAULT 0,
    taux_acceptation DECIMAL(5, 2) DEFAULT 0,
    date_maj TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (
        institution,
        date_stat,
        user_id
    )
);

CREATE INDEX idx_stats_date ON public.stats_quotidiennes (date_stat DESC);

CREATE INDEX idx_stats_institution ON public.stats_quotidiennes (institution);

CREATE INDEX idx_stats_user ON public.stats_quotidiennes (user_id);

-- =====================================================================
-- FONCTIONS HELPER
-- =====================================================================

-- Fonction : Calculer ratio d'endettement automatiquement
CREATE OR REPLACE FUNCTION calculate_ratio_endettement()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.revenu_mensuel > 0 THEN
        NEW.ratio_endettement := (NEW.charges_mensuelles / NEW.revenu_mensuel * 100)::DECIMAL(5, 2);
    ELSE
        NEW.ratio_endettement := 0;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_calculate_ratio
BEFORE INSERT OR UPDATE ON public.demandes_credit
FOR EACH ROW
EXECUTE FUNCTION calculate_ratio_endettement();

-- Fonction : Mettre à jour date_derniere_connexion
CREATE OR REPLACE FUNCTION update_last_login()
RETURNS TRIGGER AS $$
BEGIN
    NEW.date_derniere_connexion := CURRENT_TIMESTAMP;
    NEW.nb_connexions_total := COALESCE(NEW.nb_connexions_total, 0) + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- VUES POUR FACILITER LES REQUÊTES
-- =====================================================================

-- Vue : Demandes avec informations agent
CREATE OR REPLACE VIEW view_demandes_avec_agent AS
SELECT
    d.*,
    u.nom_complet AS agent_nom,
    u.institution AS agent_institution,
    a.code_agent,
    a.demandes_traitees_total
FROM public.demandes_credit d
    LEFT JOIN public.agents a ON d.agent_id = a.id
    LEFT JOIN public.users u ON a.user_id = u.id;

-- Vue : Statistiques par institution
CREATE OR REPLACE VIEW view_stats_institution AS
SELECT
    institution,
    COUNT(DISTINCT id) AS total_demandes,
    COUNT(
        DISTINCT CASE
            WHEN decision = 'ACCORDÉ' THEN id
        END
    ) AS nb_accordees,
    COUNT(
        DISTINCT CASE
            WHEN decision = 'REFUSÉ' THEN id
        END
    ) AS nb_refusees,
    COUNT(
        DISTINCT CASE
            WHEN decision = 'ÉTUDE APPROFONDIE' THEN id
        END
    ) AS nb_etude,
    ROUND(
        COUNT(
            DISTINCT CASE
                WHEN decision = 'ACCORDÉ' THEN id
            END
        ) * 100.0 / COUNT(DISTINCT id),
        2
    ) AS taux_acceptation,
    ROUND(AVG(score_ml), 2) AS score_moyen,
    SUM(montant_demande) AS total_montant_demande,
    SUM(montant_accorde) AS total_montant_accorde
FROM public.demandes_credit
GROUP BY
    institution;

-- Vue : Performance agents
CREATE OR REPLACE VIEW view_performance_agents AS
SELECT
    a.id,
    u.nom_complet,
    u.institution,
    a.code_agent,
    a.demandes_traitees_total,
    a.demandes_acceptees,
    a.demandes_refusees,
    ROUND(
        a.demandes_acceptees * 100.0 / NULLIF(a.demandes_traitees_total, 0),
        2
    ) AS taux_acceptation,
    a.score_performance,
    COUNT(DISTINCT d.id) AS demandes_en_cours,
    ROUND(AVG(d.score_ml), 2) AS score_ml_moyen
FROM public.agents a
    LEFT JOIN public.users u ON a.user_id = u.id
    LEFT JOIN public.demandes_credit d ON a.id = d.agent_id
    AND d.statut != 'archivee'
GROUP BY
    a.id,
    u.nom_complet,
    u.institution,
    a.code_agent,
    a.demandes_traitees_total,
    a.demandes_acceptees,
    a.demandes_refusees,
    a.score_performance;

-- =====================================================================
-- ROW LEVEL SECURITY (RLS) — Optionnel mais recommandé
-- =====================================================================

-- Activer RLS sur les tables sensibles
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

ALTER TABLE public.demandes_credit ENABLE ROW LEVEL SECURITY;

ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- Politique : Les agents ne voient que leurs propres demandes
CREATE POLICY "agents_see_own_demandes" ON public.demandes_credit FOR
SELECT USING (
        auth.uid () = user_id
        OR auth.uid () IN (
            SELECT id
            FROM public.users
            WHERE
                role = 'admin'
        )
    );

-- Politique : Les agents ne voient que leurs sessions
CREATE POLICY "users_see_own_sessions" ON public.sessions FOR
SELECT USING (
        auth.uid () = user_id
        OR auth.uid () IN (
            SELECT id
            FROM public.users
            WHERE
                role = 'admin'
        )
    );

-- =====================================================================
-- DONNÉES DE TEST (optionnel)
-- =====================================================================

-- Insérer un agent de test
INSERT INTO
    public.users (
        email,
        password_hash,
        nom_complet,
        institution,
        role,
        actif
    )
VALUES (
        'olivier@microfinance.cm',
        '$2b$12$5YUaoHFV0p2US5V5xgGSy../OJ180pgtokscwKZgh7yICHnZIyvz.', -- bcrypt hash de "password123"
        'KOM Olivier',
        'Microfinance XYZ',
        'agent',
        TRUE
    )
ON CONFLICT (email) DO
UPDATE
SET
    password_hash = EXCLUDED.password_hash,
    actif = TRUE;

INSERT INTO
    public.agents (
        id,
        user_id,
        code_agent,
        competences,
        demandes_traitees_total
    )
SELECT id, id, 'AGT-001', 'Scoring, Analyse financière', 45
FROM public.users
WHERE
    email = 'olivier@microfinance.cm'
ON CONFLICT DO NOTHING;

-- =====================================================================
-- RÉSUMÉ DES TABLES
-- =====================================================================
/*
✅ TABLES CRÉÉES :
1. users — Authentification agents (email, password_hash, role)
2. agents — Profil détaillé agents (code, compétences, performances)
3. demandes_credit — Demandes complètes avec scores ML
4. demandes_historique — Audit trail (qui a modifié quoi, quand)
5. sessions — Suivi des connexions
6. stats_quotidiennes — Cache pour dashboard

✅ VUES :
• view_demandes_avec_agent — Demandes + agent info
• view_stats_institution — Stats agrégées par institution
• view_performance_agents — Performance des agents

✅ FONCTIONS :
• calculate_ratio_endettement() — Calcul auto du ratio
• update_last_login() — Mise à jour connexion

✅ SÉCURITÉ :
• RLS activé sur tables sensibles
• Email format vérifié (constraint)
• UUID pour tous les IDs
• Indexes pour performances

À FAIRE APRÈS CRÉATION :
1. Configurer l'authentification Supabase (Auth providers)
2. Mettre à jour .streamlit/secrets.toml avec clés Supabase
3. Exécuter les migrations de données (si migration depuis SQLite)
4. Tester les connexions dans l'app Streamlit
*/
-- SQLBook: Code
-- =====================================================================
-- CRÉATION DES UTILISATEURS DE L'ÉQUIPE
-- Mot de passe commun : password123
-- =====================================================================

INSERT INTO
    public.users (
        email,
        password_hash,
        nom_complet,
        institution,
        role,
        actif
    )
VALUES (
        'andy@microfinance.cm',
        '$2b$12$5YUaoHFV0p2US5V5xgGSy../OJ180pgtokscwKZgh7yICHnZIyvz.',
        'Andy',
        'Microfinance XYZ',
        'agent',
        TRUE
    ),
    (
        'aristide@microfinance.cm',
        '$2b$12$5YUaoHFV0p2US5V5xgGSy../OJ180pgtokscwKZgh7yICHnZIyvz.',
        'Aristide',
        'Microfinance XYZ',
        'agent',
        TRUE
    ),
    (
        'marie@microfinance.cm',
        '$2b$12$5YUaoHFV0p2US5V5xgGSy../OJ180pgtokscwKZgh7yICHnZIyvz.',
        'Marie',
        'Microfinance XYZ',
        'agent',
        TRUE
    )
ON CONFLICT (email) DO
UPDATE
SET
    password_hash = EXCLUDED.password_hash,
    nom_complet = EXCLUDED.nom_complet,
    institution = EXCLUDED.institution,
    role = EXCLUDED.role,
    actif = EXCLUDED.actif;

INSERT INTO
    public.agents (
        id,
        user_id,
        code_agent,
        competences,
        demandes_traitees_total
    )
SELECT id, id, code_agent, 'Scoring, Analyse financière', 0
FROM (
        VALUES (
                'andy@microfinance.cm', 'AGT-ANDY'
            ), (
                'aristide@microfinance.cm', 'AGT-ARISTIDE'
            ), (
                'marie@microfinance.cm', 'AGT-MARIE'
            )
    ) AS nouveaux_agents (email, code_agent)
    JOIN public.users ON public.users.email = nouveaux_agents.email
ON CONFLICT DO NOTHING;