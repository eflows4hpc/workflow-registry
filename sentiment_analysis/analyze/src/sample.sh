runcompss sentiment_analysis.py sample_resources/dataset-1MB.json parsoda sample_resources/emoji.json output.json && \
echo "
    ###### OUTPUT ######
" && \
cat output.json && \
rm -f output.json parsoda_report.csv