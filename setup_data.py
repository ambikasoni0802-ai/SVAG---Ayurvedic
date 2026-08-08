"""
Ye script sirf ek baar chalao (terminal me), ya app.py ke upar likh sakte ho.
Ye Ayurvedic data GitHub se laata hai.
"""
import subprocess
import os

if not os.path.exists("Datasets/Ayurveda"):
    subprocess.run(["git", "clone", "https://github.com/gita/Datasets.git"])
    subprocess.run(["rm", "-rf", "Datasets/chanakya", "Datasets/srimad-bhagavatam",
                     "Datasets/Vectorise_Script", "Datasets/README.md"])

if not os.path.exists("herb-database"):
    subprocess.run(["git", "clone", "https://github.com/sciencewithsaucee-sudo/herb-database.git"])

print("Data ready hai!")
