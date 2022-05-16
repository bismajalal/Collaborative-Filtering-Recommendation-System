from jproperties import Properties

#open file to write values
p = Properties()
with open("config.properties", "rb") as f:
    p.load(f, "utf-8")

p["dataset"] = "F:\Downloads\DL\Recommendation System\Data\\rating_small.csv"
p["train"] = "F:\Downloads\DL\Recommendation System\Data\\train.csv"
p["test"] = "F:\Downloads\DL\Recommendation System\Data\\test.csv"
p["path"] = "F:\Downloads\DL\Recommendation System\Code\\my-model"

with open("config.properties", "wb") as f:
    p.store(f, encoding="utf-8")
