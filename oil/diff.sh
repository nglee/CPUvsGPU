for i in $(seq -f "%02g" 0 7)
do
    diff "$i.oil.jpg" "$i.oil.omp.jpg"
    diff "$i.oil.jpg" "$i.oil.cuda.jpg"
    diff "$i.oil.cuda.jpg" "$i.oil.cuda.shared.jpg"
    diff "$i.oil.cuda.jpg" "$i.oil.cuda.second.jpg"
done
