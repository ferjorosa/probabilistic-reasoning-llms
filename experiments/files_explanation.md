Vamos a tener 3 tipos archivos:

networks.parquet -> contiene las redes que se van a considerar con un id unico (model_id)
queries.parquet -> contiene las queries
llm_results.parquet -> los resultados donde cada columna es un modelo, podemos tener run, si queremos considerar la misma combinacion multiples veces (estocasticidad)