"""SemBench's own Palimpzest programs for the MMQA queries, copied from the SemBench repository
(`src/scenario/mmqa/runner/palimpzest_runner/palimpzest_runner.py`, github.com/sembench/sembench), one
function per query: qN(pz_config, data_dir) -> the program's output (q4 returns SemBench's post-processed
{"results", "execution_stats"}). `data_dir` is the suite's sf_200 directory (CSV files and images/). Run by
../palimpzest_exec.py in Palimpzest's interpreter.
"""
import os

import palimpzest as pz
import pandas as pd


def load_data(data_dir, name, **kw):
    return pd.read_csv(os.path.join(data_dir, name), **kw)


def q1(cfg, data_dir):
    table_df = load_data(data_dir, "ben_piazza.csv")
    text_df = load_data(data_dir, "ben_piazza_text_data.csv")
    joined_df = table_df.merge(text_df, left_on="Title", right_on="title", how="left").fillna("")
    ds = pz.MemoryDataset(id="ben_piazza_joined", vals=joined_df)
    ds = ds.sem_map([{"name": "director", "type": str, "desc": "Extract the director name from the movie description."}], depends_on=["text"])
    ds = ds.filter(lambda row: row["Role"] == "Bob Whitewood")
    ds = ds.project(["director"])
    return ds.run(cfg)


def _images(data_dir):
    return pz.ImageFileDataset(id="images", path=os.path.join(data_dir, "images"))


def q2a(cfg, data_dir):
    pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=load_data(data_dir, "ap_warrior.csv"))
    pz_table = pz_table.sem_join(_images(data_dir),
                                 "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack.",
                                 depends_on=["Track", "contents"])
    pz_table = pz_table.project(["ID", "filename"])
    return pz_table.run(cfg)


def q2b(cfg, data_dir):
    pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=load_data(data_dir, "ap_warrior.csv"))
    pz_table = pz_table.sem_join(_images(data_dir),
                                 "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack.",
                                 depends_on=["Track", "contents"])
    pz_table = pz_table.sem_map([{"name": "color", "type": str, "desc": "The color of the logo in the image"}], depends_on=["contents"])
    pz_table = pz_table.project(["ID", "filename", "color"])
    return pz_table.run(cfg)


def _text_filter(cfg, data_dir, prompt):
    pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=load_data(data_dir, "lizzy_caplan_text_data.csv"))
    pz_text = pz_text.sem_filter(prompt, depends_on=["title", "text"])
    pz_text = pz_text.project(["title"])
    return pz_text.run(cfg)


def q3a(cfg, data_dir):
    return _text_filter(cfg, data_dir, "Determine if a movie is a comedy movie given their description.")


def q3f(cfg, data_dir):
    return _text_filter(cfg, data_dir, "Determine if a movie is a romantic comedy given their description.")


def q4(cfg, data_dir):
    text_df = load_data(data_dir, "lizzy_caplan_text_data.csv")
    target_values = ["Orange County", "Mean Girls", "Love Is the Drug", "Crashing", "Cloverfield", "My Best Friend's Girl",
                     "Crossing Over", "Hot Tub Time Machine", "The Last Rites of Ransom Pride", "127 Hours", "High Road",
                     "Save the Date", "Bachelorette", "3, 2, 1... Frankie Go Boom", "Queens of Country", "Item 47",
                     "The Interview", "The Night Before", "Now You See Me 2", "Allied", "The Disaster Artist", "Extinction",
                     "The People We Hate at the Wedding", "Cobweb"]
    text_df = text_df[text_df["title"].isin(target_values)]
    pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)
    pz_text = pz_text.sem_map([{"name": "genres", "type": str, "desc": "The genres of the movie, separated by commas"}], depends_on=["text"])
    pz_text = pz_text.project(["title", "genres"])
    output = pz_text.run(cfg)
    output_df = output.to_df()
    expanded = []
    for _, row in output_df.iterrows():
        genres = [g.lower().strip() for g in row["genres"].split(",")] if isinstance(row["genres"], str) else []
        for genre in genres:
            expanded.append({"genre": genre, "title": row["title"]})
    df_expanded = pd.DataFrame(expanded)
    table = df_expanded.groupby("genre")["title"].apply(lambda x: ", ".join(x)).reset_index() if len(df_expanded) else pd.DataFrame(columns=["genre", "title"])
    table.rename(columns={"title": "movies_in_genre"}, inplace=True)
    return {"results": table, "execution_stats": output.execution_stats}


def q5(cfg, data_dir):
    text_df = load_data(data_dir, "lizzy_caplan_text_data.csv", sep=",", quotechar='"')
    target_values = ["Love Is the Drug", "Crashing", "Cloverfield", "My Best Friend's Girl", "Hot Tub Time Machine",
                     "The Last Rites of Ransom Pride", "Save the Date", "Bachelorette", "3, 2, 1... Frankie Go Boom",
                     "Queens of Country", "Item 47", "The Night Before", "Now You See Me 2", "Allied", "Extinction", "Cobweb"]
    text_df = text_df[text_df["title"].isin(target_values)]
    pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)
    pz_text = pz_text.sem_agg(col={"name": "actor", "type": str, "description": "The name of the actor"},
                              agg="Who has played a role in all the movies listed in the table given their descriptions? Simply give the name of the actor.",
                              depends_on=["title", "text"])
    return pz_text.run(cfg)


def _airport(cfg, data_dir, prompt):
    pz_table = pz.MemoryDataset(id="tampa_airport", vals=load_data(data_dir, "tampa_international_airport.csv"))
    pz_table = pz_table.sem_filter(prompt, depends_on=["Airlines", "Destinations"])
    pz_table = pz_table.project(["Airlines"])
    return pz_table.run(cfg)


def q6a(cfg, data_dir):
    return _airport(cfg, data_dir, "Given destinations of an airline, the airline has flights to Frankfurt.")


def q6b(cfg, data_dir):
    return _airport(cfg, data_dir, "Given destinations of an airline, the airline has flights to Germany.")


def q6c(cfg, data_dir):
    return _airport(cfg, data_dir, "Given destinations of an airline, the airline has flights to Europe.")


def q7(cfg, data_dir):
    pz_table = pz.MemoryDataset(id="tampa_airport", vals=load_data(data_dir, "tampa_international_airport.csv"))
    pz_table = pz_table.sem_join(_images(data_dir), "The image shows the airline logo.", depends_on=["Airlines", "contents"])
    pz_table = pz_table.project(["Airlines", "filename"])
    return pz_table.run(cfg)
