# ==========================================================
# Exhaustive LR model using all SNP combinations
# + SEXO + IDADE + ETNIA covariates
# ==========================================================


import pandas as pd
import numpy as np
import os
import time

from itertools import combinations


from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score
)


from sklearn.pipeline import Pipeline


from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression



# ==========================================================
# Configurações
# ==========================================================


input_dir = "data/processed_data"


output_dir = "data/processed_data/lr_model"


os.makedirs(
    output_dir,
    exist_ok=True
)



N_SPLITS = 5

N_REPEATS = 5


RANDOM_STATE = 42




# ==========================================================
# Cross validation
# ==========================================================


cv = RepeatedStratifiedKFold(

    n_splits=N_SPLITS,

    n_repeats=N_REPEATS,

    random_state=RANDOM_STATE

)




# ==========================================================
# Função principal
# ==========================================================


def run_lr_combinations(

        filepath,

        inheritance_model

):


    start = time.time()


    print("\n")
    print("="*80)
    print(
        f"MODELO DE HERANÇA: {inheritance_model}"
    )
    print("="*80)



    # --------------------------
    # Dados
    # --------------------------

    df = pd.read_csv(filepath)



    y = df["ELISA"]



    X = df.drop(
        columns=["ELISA"]
    )



    # --------------------------
    # Covariáveis
    # --------------------------

    covariates = [

        "SEXO",

        "IDADE",

        "ETNIA"

    ]


    covariates = [

        c for c in covariates

        if c in X.columns

    ]



    # SNPs disponíveis

    snps = [

        c for c in X.columns

        if c not in covariates

    ]



    print(
        "Covariáveis:",
        covariates
    )


    print(
        "Número de SNPs:",
        len(snps)
    )



    combinations_list = []



    for size in range(

        1,

        len(snps)+1

    ):


        combinations_list.extend(

            combinations(

                snps,

                size

            )

        )



    total = len(
        combinations_list
    )



    print(
        f"Total combinações: {total}"
    )



    results = []



    # ======================================================
    # Loop combinações
    # ======================================================


    for i, combo in enumerate(

        combinations_list,

        start=1

    ):


        print(

            f"{inheritance_model} | "
            f"LR | "
            f"rodando {i}/{total}"
            f" combinações"

        )



        selected = (

            covariates

            +

            list(combo)

        )


        X_model = X[
            selected
        ]



        # ----------------------
        # preprocessing
        # ----------------------


        numeric_features = [

            c for c in selected

            if c != "ETNIA"

        ]


        categorical_features = (

            ["ETNIA"]

            if "ETNIA" in selected

            else []

        )



        preprocess = ColumnTransformer(

            transformers=[


                (

                    "num",

                    Pipeline([


                        (

                            "imputer",

                            SimpleImputer(

                                strategy="median"

                            )

                        ),


                        (

                            "scaler",

                            StandardScaler()

                        )

                    ]),

                    numeric_features

                ),



                (

                    "cat",

                    Pipeline([


                        (

                            "imputer",

                            SimpleImputer(

                                strategy="most_frequent"

                            )

                        ),


                        (

                            "encoder",

                            OneHotEncoder(

                                handle_unknown="ignore"

                            )

                        )

                    ]),

                    categorical_features

                )

            ]

        )




        model = Pipeline([


            (

                "preprocess",

                preprocess

            ),


            (

                "classifier",

                LogisticRegression(

                    max_iter=5000,

                    random_state=RANDOM_STATE

                )

            )

        ])




        auc = cross_val_score(

            model,

            X_model,

            y,

            cv=cv,

            scoring="roc_auc",

            n_jobs=-1

        )



        results.append({


            "MODELO":

            inheritance_model,


            "ESTIMADOR":

            "LR",


            "COMBINACAO":

            " + ".join(combo),


            "NUMERO DE SNPS":

            len(combo),


            "AUC MEDIO":

            auc.mean(),


            "DESVIO PADRAO":

            auc.std()

        })




    result_df = pd.DataFrame(
        results
    )


    # ordenar melhor resultado

    result_df = (

        result_df

        .sort_values(

            "AUC MEDIO",

            ascending=False

        )

    )



    result_df.to_csv(

        f"{output_dir}/"
        f"{inheritance_model}_LR_results.csv",

        index=False

    )



    print(

        f"{inheritance_model} finalizado "
        f"{(time.time()-start)/60:.2f} minutos"

    )


    return result_df





# ==========================================================
# Rodar os 3 modelos
# ==========================================================


all_results = []



for inheritance in [

    "additive",

    "dominant",

    "recessive"

]:


    filepath = (

        f"{input_dir}/"

        f"{inheritance}_imputed.csv"

    )


    result = run_lr_combinations(

        filepath,

        inheritance

    )


    all_results.append(
        result
    )





# ==========================================================
# Unir tudo
# ==========================================================


final_results = pd.concat(

    all_results,

    ignore_index=True

)



final_results = (

    final_results

    .sort_values(

        "AUC MEDIO",

        ascending=False

    )

)



final_results.to_csv(

    f"{output_dir}/all_LR_results.csv",

    index=False

)



print("\n")
print("="*80)
print("FINALIZADO")
print("="*80)


print(
    final_results.head(20)
)