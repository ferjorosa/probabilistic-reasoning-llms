La idea principal es que creemos el codigo para guardar los experimentos primero en un base de datos local (SQLite) y luego en la base de datos de Aily

Si los experimentos son muchos pues tendremos que usar el aily-brain para no usar nuestro propio dinero en ello.

----------

Antes de esto, la idea es obtener una estimacion del numero de redes que vamos a probar, que parametros etc. Hay que revisar el codigo de Bojan

Luego, yo crearia un dataset estable que mas tarde subiriamos a HuggingFace

Cuando tengamos claro que redes vamos a probar y que queries, entonces ya podemos hacer el codigo para ejecutar esos experimentos de tal forma que carguen el dataset que hemos generado y que como resultado generaria otro dataset que tambien compartiriamos en HuggingFace para que la gente lo revise. Deberia contener a poder ser 