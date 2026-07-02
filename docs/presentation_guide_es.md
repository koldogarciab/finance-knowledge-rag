# Guía para presentar el proyecto

## Explicación breve de 30 segundos

> He desarrollado un RAG financiero completamente local que integra información de PDF, DOCX, CSV, JSON y Markdown. El sistema combina embeddings MiniLM con BM25 mediante weighted reciprocal rank fusion, y utiliza Ollama para generar respuestas solo a partir de evidencia seleccionada. Añadí validación de citas, controles de consistencia financiera, fallback seguro y abstención previa cuando la información no existe. En la evaluación final respondió 30 de 30 preguntas sustentadas, rechazó correctamente 10 de 10 no sustentadas y obtuvo un 95,44% de groundedness medio.

## Presentación de 5–7 minutos

### 1. Problema

> En finanzas la información está distribuida entre actuals, políticas, KPIs, informes y actas. Un buscador semántico simple puede recuperar algo parecido, pero eso no garantiza que la respuesta sea correcta, que use el periodo adecuado o que no invente cifras.

### 2. Corpus

> Creé un corpus financiero sintético y reproducible en cinco formatos. Extraje 305 registros y los convertí en 316 chunks token-aware, conservando metadatos de página, sección, mes, departamento y granularidad.

### 3. Retriever

> Empecé con un baseline denso basado en MiniLM. Después añadí BM25 porque las preguntas financieras contienen muchas coincidencias exactas: nombres de KPIs, meses, thresholds y departamentos. Combiné ambos rankings con weighted RRF. Los parámetros se eligieron con 10 preguntas de desarrollo y el benchmark formal de 30 preguntas quedó congelado.

### 4. Mejora cuantitativa

> El MRR pasó de 0,7117 a 0,8107 y el Hit@5 de 73,33% a 90%. El rango medio bajó de 18,3 a 3,97. La mejora fue especialmente fuerte en PDF, donde el híbrido alcanzó un MRR de 1,0.

### 5. Generación local

> Usé Ollama con qwen3.5:4b para no depender de una API ni pagar tokens. El modelo recibe únicamente los chunks aprobados por el grounding gate.

### 6. Seguridad

> La parte más importante no es el chat, sino los safeguards. El sistema valida citas, comprueba números y polaridad financiera, repara respuestas inválidas y utiliza un fallback extractivo si la evidencia es suficiente. Cuando la información no existe, se abstiene antes de llamar al modelo.

### 7. Evaluación end-to-end

> En 30 preguntas sustentadas respondió las 30, sin abstenciones incorrectas. El chunk esperado fue recuperado, enviado y citado en el 100% de los casos. Las citas válidas fueron 100% y el groundedness medio 95,44%. En 10 preguntas no sustentadas se abstuvo correctamente en las 10, evitando todas las llamadas al modelo.

### 8. Resultado final

> Tras revisión manual, 29 respuestas fueron aceptadas y una quedó como parcial conservadora, no como alucinación. Congelé la versión v5.3 para evitar seguir ajustando reglas al benchmark.

### 9. Interfaz

> Finalmente construí una interfaz Streamlit que llama directamente al pipeline evaluado. Muestra la respuesta, las citas, la fuente, el modo de generación y la latencia.

## Preguntas probables

### ¿Por qué el peso léxico es tan alto?

Porque el corpus contiene meses, departamentos, nombres de KPIs y umbrales muy concretos. BM25 aporta precisión donde la similitud semántica puede confundir documentos con estructura parecida.

### ¿Por qué weighted RRF?

Porque fusiona rankings sin exigir que las puntuaciones densas y BM25 estén en la misma escala. Es simple, interpretable y robusto.

### ¿Por qué no usar una API más potente?

El objetivo era demostrar un sistema local, reproducible y sin coste variable. El modelo 4B tiene limitaciones, pero el grounding y los fallbacks reducen el riesgo.

### ¿Qué significa groundedness?

Es la proporción de afirmaciones de la respuesta que pueden vincularse con la evidencia seleccionada. No sustituye la revisión humana, pero ayuda a detectar respuestas no sustentadas.

### ¿Por qué existe un fallback extractivo?

Porque una respuesta generada puede fallar por formato, citas o completitud aunque la evidencia correcta esté disponible. El fallback permite responder de forma segura sin inventar.

### ¿Cuál es la principal limitación?

El corpus y el benchmark son sintéticos y pequeños. En producción necesitaría más documentos, usuarios reales, control de accesos, freshness, auditoría y monitorización.

### ¿Qué mejorarías después?

- reranking aprendido;
- expansión del benchmark;
- evaluación con expertos financieros;
- control de acceso por documento;
- ingestion incremental;
- modelo local mayor o cuantizado;
- métricas de confianza y observabilidad.

## Mensaje final

> El valor del proyecto no está solo en responder preguntas, sino en demostrar todo el ciclo: datos, chunking, embeddings, retrieval, tuning, benchmark congelado, generación local, evaluación, control de alucinaciones, interfaz y documentación.
