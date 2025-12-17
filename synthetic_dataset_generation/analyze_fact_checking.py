import pandas as pd
import argparse
import ast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fact_checking_csv", type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.fact_checking_csv)

    new_df = {
        "input_text": [],
        "claims": [],
        "claim_labels": [],
        "verdict_reasons": [],
    }
    # input_text	claims	claim_labels	verdict_reasons
    for index, row in df.iterrows():
        input_text = row["input_text"]
        claims = ast.literal_eval(row["claims"])
        claim_labels = ast.literal_eval(row["claim_labels"])
        verdict_reasons = ast.literal_eval(row["verdict_reasons"])

        assert len(claims) == len(claim_labels) == len(verdict_reasons)
        for claim, label, reason in zip(claims, claim_labels, verdict_reasons):
            new_df["input_text"].append(input_text)
            new_df["claims"].append(claim)
            new_df["claim_labels"].append(label)
            new_df["verdict_reasons"].append(reason)
        
    new_df = pd.DataFrame(new_df)
    new_df.to_excel(args.fact_checking_csv.replace(".csv", ".xlsx"), index=False, engine="xlsxwriter")
    