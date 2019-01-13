from Shapenet import get_shapenet_metadata

categories, split = get_shapenet_metadata("/home/krabec/data/ShapeNet")

print("here")
test = [0] * 55
val = [0] *55
train = [0] *55
print(len(categories))
print(len(split))

for key in split.keys():
    cat = categories[key]
    spl = split[key]
    if spl == 2:
        test[cat] +=1
print(test)