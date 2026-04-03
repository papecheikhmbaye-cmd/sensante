# -*- coding: utf-8 -*-

"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd


def main():
    # ===== CHARGER LES DONNEES =====
    try:
        df = pd.read_csv("../data/patients_dakar.csv")
    except FileNotFoundError:
        print("Erreur : fichier 'data/patients_dakar.csv' introuvable.")
        print("Verifier que le fichier existe dans le dossier data/")
        return

    # ===== PREMIERS APERÇUS =====
    print("=" * 50)
    print("SENSANTE - Exploration du dataset")
    print("=" * 50)

    # Dimensions du dataset
    print(f"Nombre de patients : {len(df)}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print(f"Colonnes : {list(df.columns)}")

    # Aperçu des 5 premières lignes
    print("\n--- 5 premiers patients ---")
    print(df.head())

    # ===== STATISTIQUES DE BASE =====
    print("\n--- Statistiques descriptives ---")
    print(df.describe().round(2))

    # ===== REPARTITION DES DIAGNOSTICS =====
    print("\n--- Répartition des diagnostics ---")
    if "diagnostic" in df.columns:
        diag_counts = df["diagnostic"].value_counts()
        for diag, count in diag_counts.items():
            pct = count / len(df) * 100
            print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")
    else:
        print("Colonne 'diagnostic' introuvable.")

    # ===== REPARTITION PAR REGION =====
    print("\n--- Répartition par region (top 5) ---")
    if "region" in df.columns:
        region_counts = df["region"].shape[0] # Correction mineure pour cohérence
        region_counts = df["region"].value_counts().head(5)
        for region, count in region_counts.items():
            print(f"{region:15s} : {count:3d} patients")
    else:
        print("Colonne 'region' introuvable.")

    # ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
    print("\n--- Temperature moyenne par diagnostic ---")
    if "diagnostic" in df.columns and "temperature" in df.columns:
        temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
        for diag, temp in temp_by_diag.items():
            print(f"{diag:12s} : {temp:.1f} C")
    else:
        print("Colonnes necessaires manquantes (diagnostic, temperature).")

    # ==========================================================
    # EXERCICE 1 : ANALYSE CROISÉE (Sexe et Diagnostic)
    # ==========================================================
    print("\n--- Analyse croisée : Sexe et Diagnostic ---")
    if "sexe" in df.columns and "diagnostic" in df.columns:
        # Utilisation de groupby().size() comme demandé dans l'exercice
        analyse_croisee = df.groupby(["sexe", "diagnostic"]).size()
        print(analyse_croisee)
    else:
        print("Colonnes 'sexe' ou 'diagnostic' manquantes pour l'analyse croisée.")

    print("\n" + "=" * 50)
    print("Exploration terminee")
    print("Prochain lab : entrainer un modele ML")
    print("=" * 50)


if __name__ == "__main__":
    main()