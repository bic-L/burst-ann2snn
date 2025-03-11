declare -A classes=(
    ["Airplane"]=4
    ["Bag"]=2
    ["Cap"]=2
    ["Car"]=4
    ["Chair"]=4
    ["Earphone"]=3
    ["Guitar"]=3
    ["Knife"]=2
    ["Lamp"]=4
    ["Laptop"]=2
    ["Motorbike"]=6
    ["Mug"]=2
    ["Pistol"]=3
    ["Rocket"]=3
    ["Skateboard"]=3
    ["Table"]=3
)

# 遍历类别
for class in "${!classes[@]}"; do
    echo "Training on class: $class"
    python train_segmentation.py --class_choice "$class" --lr 0.0001
done

echo "All training processes have been completed."