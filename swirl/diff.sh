for i in $(seq -f "%02g" 0 7)
do
    diff "$i.twisted.jpg" "$i.twisted.omp.jpg"
    diff "$i.twisted.jpg" "$i.twisted.cuda.jpg"
done
