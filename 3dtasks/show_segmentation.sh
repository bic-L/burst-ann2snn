#/bin/sh

# Define the classes and their corresponding quntities
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

# Iterate over the classes
for class in "${!classes[@]}"; do
    echo "Testing on class: $class"
    python show_seg.py --class_choice "$class"  --model ../checkpoints/3dsegmentation/3dseg_checkpoints/$class/best_seg_model.pth
done

echo "All training processes have been completed."
