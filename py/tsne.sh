#for PERP in 2 5 10 20 30 40 50
for ITER in 10000 20000
do
	echo "Trying $ITER"
	python bhtsne.py -i fqsp_embed.tsv -o fqsp_tsne_i$ITER.tsv -p 40 -m $ITER
done

