from sklearn.impute import SimpleImputer

import pandas as pd
import re
import os



# ==========================================================
# Leitura dos dados
# ==========================================================

genotypes = pd.read_excel(
    "data/genotypes.xlsx"
)



# remover indivíduos sem ID

genotypes = genotypes.dropna(
    subset=["ID"]
)



# ==========================================================
# Selecionar variáveis do modelo
# ==========================================================

model_data = genotypes.loc[
    :,
    [
        "SEXO",
        "IDADE",
        "ETNIA",
        "ELISA_IgG",
        "IL17_rs2275913G_A",
        "IL17_rs4711998A_G",
        "FAS670_rs1800682G_A",
        "FOXP3_rs3761548G_T",
        "ACE1",
        "IL4_rs2243250T_C",
        "IL10_rs1800896T_C",
        "ACE2_rs2285666G_A",
        "ACE2_rs73635825A_G",
        "TMPRSS2_rs12329760C_T"
    ]
].copy()



# limpar nomes das colunas

model_data.columns = (
    model_data.columns
    .str.strip()
)



# ==========================================================
# Remover MBL
# ==========================================================

model_data = model_data.loc[
    :,
    ~model_data.columns.str.startswith("MBL")
]



# ==========================================================
# Recodificação das variáveis clínicas
# ==========================================================


# Sexo

model_data["SEXO"] = model_data["SEXO"].map(
    {
        "Masculino": 1,
        "Feminino": 0
    }
)



# Etnia
#
# Caso já esteja 1-6 mantém.
# Caso seja texto, converte para categorias.

model_data

if model_data["ETNIA"].dtype == "object":

    model_data["ETNIA"] = (
        model_data["ETNIA"]
        .astype("category")
        .cat.codes
        + 1
    )

else:

    model_data["ETNIA"] = (
        model_data["ETNIA"]
        .astype("Int64")
    )



# ELISA

model_data["ELISA"] = model_data["ELISA_IgG"].map(
    {
        "Reagente": 1,
        "Não Reagente": 0
    }
)



model_data = model_data.drop(
    columns=["ELISA_IgG"]
)



# ==========================================================
# Limpeza dos genótipos
# ==========================================================

def clean_genotype(x):

    if pd.isna(x):
        return x

    return (
        str(x)
        .strip()
        .replace(" ", "")
        .upper()
    )



# somente SNPs

snp_columns = [

    c for c in model_data.columns

    if "rs" in c

]



for col in snp_columns:

    model_data[col] = (
        model_data[col]
        .apply(clean_genotype)
    )



# ==========================================================
# Mapa ACE1
# ==========================================================

ACE1_maps = {


    "additive": {

        "II":0,
        "ID":1,
        "DI":1,
        "DD":2

    },


    "dominant": {

        "II":0,
        "ID":1,
        "DI":1,
        "DD":1

    },


    "recessive": {

        "II":0,
        "ID":0,
        "DI":0,
        "DD":1

    }

}




# ==========================================================
# Criar mapas SNP automaticamente
# ==========================================================

def create_snp_map(column):


    match = re.search(

        r"rs\d+([ACGT])_([ACGT])",

        column

    )


    if match is None:

        return None



    wt = match.group(1)

    mut = match.group(2)



    return {


        "additive": {


            wt+wt:0,

            wt+mut:1,

            mut+wt:1,

            mut+mut:2,


            # casos com uma letra

            wt:0,

            mut:2

        },



        "dominant": {


            wt+wt:0,

            wt+mut:1,

            mut+wt:1,

            mut+mut:1,


            wt:0,

            mut:1

        },



        "recessive": {


            wt+wt:0,

            wt+mut:0,

            mut+wt:0,

            mut+mut:1,


            wt:0,

            mut:1

        }

    }





# ==========================================================
# Criar modelos genéticos
# ==========================================================

def build_genetic_model(

        df,

        model_type

):


    data = df.copy()



    # SNPs

    for snp in snp_columns:


        mapping = create_snp_map(snp)


        if mapping is None:

            continue



        data[snp] = (

            data[snp]

            .map(

                mapping[model_type]

            )

        )




    # ACE1

    if "ACE1" in data.columns:


        data["ACE1"] = (

            data["ACE1"]

            .astype(str)

            .str.strip()

            .str.upper()

            .map(

                ACE1_maps[model_type]

            )

        )



    return data





# ==========================================================
# Criar os três modelos
# ==========================================================

additive = build_genetic_model(

    model_data,

    "additive"

)


dominant = build_genetic_model(

    model_data,

    "dominant"

)


recessive = build_genetic_model(

    model_data,

    "recessive"

)





# ==========================================================
# Imputação
# ==========================================================

output_dir = "data/processed_data"


os.makedirs(

    output_dir,

    exist_ok=True

)




def impute_and_save(

        df,

        filename

):


    imputer = SimpleImputer(

        strategy="median"

    )



    columns = df.columns



    df_imputed = pd.DataFrame(

        imputer.fit_transform(df),

        columns=columns,

        index=df.index

    )



    df_imputed.to_csv(

        f"{output_dir}/{filename}",

        index=False

    )



    return df_imputed





# ==========================================================
# Salvar datasets finais
# ==========================================================


additive_imputed = impute_and_save(

    additive,

    "additive_imputed.csv"

)



dominant_imputed = impute_and_save(

    dominant,

    "dominant_imputed.csv"

)



recessive_imputed = impute_and_save(

    recessive,

    "recessive_imputed.csv"

)




print("\nArquivos salvos em:")
print(output_dir)



print("\nDimensões:")

print(
    "Additive:",
    additive_imputed.shape
)

print(
    "Dominant:",
    dominant_imputed.shape
)

print(
    "Recessive:",
    recessive_imputed.shape
)



print("\nColunas finais:")

print(
    additive_imputed.columns.tolist()
)