from datasets import load_dataset

ds = load_dataset("Kyrmasch/sKQuAD")

ds.save_to_disk("/home/cyberdemon/Desktop/Datathon")
print(ds)
