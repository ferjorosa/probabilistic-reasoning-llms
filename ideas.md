Lo principal es analizar como de bien responden los LLMs ante preguntas de inferencia probabilistica

Empezamos con las probabilidades y la red completamente definida y se la pasamos al modelo. El LLM tendria que resolver "a mano", tipo papel y lapiz las queries. Lo bueno de esto es que vemos que pasos va tomando y si tienen sentido. Lo malo es que claramente no es escalable

Segun chat-gpt:

With ~8–10 variables (binary), it’s manageable by hand if we’re careful (though tedious).

With ~20+ variables, the number of terms explodes; without automation, it becomes infeasible to do exactly.

At that point, an LLM without a probabilistic engine can only:

Do small toy examples exactly.

Do approximations (like reasoning qualitatively: “smoking increases probability of bronchitis, which makes dyspnea more likely”).

-----------------------------------------

Una vez tenemos los ejemplos con variables discretas, podriamos probar a hacer inferencias en redes Gaussianas, que al fin y al cabo es lo mismo que hacer inferencia sobre una Gaussiana multivariante

------------------------------------------

El objetivo es luego probar (tipo ablation test) si estos resultados son consistentes cuando no representamos las probabilidades en forma de CPT sino que lo ponemos en formato de texto

Ver si el modelo sigue un acercamiento quantitativo o cualitativo, probando a su vez diferentes tipos de prompts

-------------------------------------------

Finalmente, tenemos que comparar esto con un agente, es decir, darle al LLM herramientas y ver como diferente tipos de herramientas afectan a su performance. 

* Que pasa si le damos una calculadora.
* Que pasa si le damos acceso a codigo
* Que pasa si le damos acceso a un MCP server
* etc.

------------------------

El nivel 1 de dificultad es indicarle en el prompt que les tamos pasando una BN, con el formato y demas