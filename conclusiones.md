# Conclusiones — TP3 Clasificador de Emociones

Documento de trabajo para ir registrando resultados, observaciones y conclusiones del notebook `Luna-Marcelo-DL-TP3-CO24.ipynb`.

---

## 1. Resultados del modelo actual (baseline)

Configuración principal: balanceo con `WeightedRandomSampler`, early stopping en `val_f1`, `LEARNING_RATE = 1e-3`, imágenes 48×48 en escala de grises, augmentations (volteo horizontal + color jitter).

| Métrica | Validación |
|---------|------------|
| Accuracy | 73.3% |
| F1 macro | 65.8% |
| Mejor época | 19 (early stop en época 26) |
| Loss | 0.80 |

### Rendimiento por clase

| Clase | Precision | Recall | F1 | Observación |
|-------|-----------|--------|-----|-------------|
| alegría | 0.91 | 0.85 | 0.88 | Mejor clase; confusión menor con seriedad/tristeza |
| sorpresa | 0.70 | 0.80 | 0.74 | Buen rendimiento |
| enojo | 0.69 | 0.75 | 0.72 | Aceptable |
| seriedad | 0.66 | 0.69 | 0.68 | Confusión con tristeza y sorpresa |
| tristeza | 0.63 | 0.56 | 0.60 | Recall bajo; mucha confusión con seriedad |
| miedo | 0.59 | 0.55 | 0.57 | Confusión con sorpresa (~19%) |
| disgusto | 0.38 | 0.49 | 0.43 | Peor clase; confusión frecuente con seriedad |

El F1 macro (~66%) queda por debajo del accuracy (~73%) y del weighted F1 (~74%), lo que indica que las clases minoritarias o difíciles arrastran el rendimiento global.

---

## 2. Evolución del entrenamiento (experimentos realizados)

- **Sin balanceo de clases:** se observó sobreajuste.
- **Con balanceo:** mejoró notablemente, pero persistió overfitting residual y ruido en las curvas.
- **Learning rate reducido:** disminuyó el ruido; el sobreajuste residual continuó.
- **Early stopping (`val_f1`, paciencia 7):** el entrenamiento se detuvo alrededor de la época 26; se restauraron los pesos de la época 19 (mejor `val_f1`: 0.6582). Mejoró la matriz de confusión en disgusto (ej.: disgusto/disgusto pasó de ~0.23 a ~0.37; disgusto/seriedad bajó de ~0.38 a ~0.19).

<!-- Agregar aquí nuevos experimentos -->
<!-- Ejemplo: -->
<!-- - **64×64 + rotación + recorte:** acc __%, F1 macro __% -->

---

## 3. Patrones en la matriz de confusión

1. **Tristeza ↔ seriedad:** par más problemático. Expresiones neutras/negativas similares en imágenes de 48×48 en escala de grises.
2. **Disgusto → seriedad:** el balanceo ayuda el recall de disgusto pero baja la precisión (muchas predicciones incorrectas hacia o desde seriedad).
3. **Miedo ↔ sorpresa:** ojos y boca abiertos se confunden a baja resolución.
4. **Train acc < val acc** (~64% vs ~73% en la mejor época): comportamiento esperable con balanceo, dropout y augmentations solo en entrenamiento.

Confusiones más frecuentes (validación, modelo actual):

- Tristeza real → seriedad predicha: 106 casos (~22% de tristeza).
- Disgusto real → seriedad predicha: 34 casos (~21% de disgusto).
- Miedo real → sorpresa predicha: 14 casos (~19% de miedo).

---

## 4. Desbalance del dataset

Distribución aproximada en entrenamiento:

| Clase | Imágenes (train) |
|-------|------------------|
| alegría | ~4772 |
| seriedad | ~2524 |
| tristeza | ~1982 |
| sorpresa | ~1290 |
| disgusto | ~717 |
| enojo | ~705 |
| miedo | ~281 |

Las clases con menos ejemplos (miedo, disgusto, enojo) coinciden con los F1 más bajos, aunque el balanceo mitiga parcialmente el efecto.

---

## 5. Inferencia en imágenes propias (puntos 4 y 5)

En validación, tristeza **no** está sobre-predicha (recall 56%, precision 63%). Si en fotos propias el modelo sesga hacia tristeza u otra clase, lo más probable es **domain shift**:

| Entrenamiento / validación | Imágenes propias |
|----------------------------|------------------|
| Rostros ya alineados (`*_aligned.jpg`) | Fotos completas, otro encuadre e iluminación |
| Pipeline fijo 48×48 gris | Sin recorte o recorte Haar imperfecto |

Observaciones:

- En inferencia debe usarse siempre **`transform_val`** (sin augmentations de entrenamiento).
- El **punto 5** (detección de rostro + recorte) suele ser más confiable que el punto 4 para fotos reales.
- Conviene revisar los scores softmax: si top-1 y top-2 están muy cerca, la predicción es incierta.

Parámetros actuales de detección Haar: `SCALE_FACTOR=1.1`, `MIN_NEIGHBORS=6`, `MIN_SIZE=(30,30)`.

<!-- Registrar resultados de pruebas con imágenes propias -->
<!-- Ejemplo: -->
<!-- - Punto 4 (sin recorte): X/Y aciertos -->
<!-- - Punto 5 (con Haar): X/Y aciertos -->

---

## 6. Conclusiones generales (borrador para el informe)

- Se construyó una CNN desde cero con pipeline de preprocesamiento configurable (resize, escala de grises, normalización, augmentations, balanceo).
- El balanceo de clases y el early stopping en F1 macro fueron decisiones clave: sin balanceo había sobreajuste; con early stopping se evitó degradar el modelo en épocas tardías.
- El rendimiento global en validación (~73% accuracy) es razonable, pero el **F1 macro (~66%)** muestra que el modelo aún falla en clases minoritarias y en pares de emociones visualmente parecidas.
- El principal límite no es solo arquitectura o hiperparámetros, sino la **dificultad semántica** de distinguir tristeza/seriedad y disgusto/seriedad en imágenes de muy baja resolución.
- En imágenes externas al dataset, la calidad del **alineamiento del rostro** pesa más que un pequeño ajuste de learning rate: el modelo fue entrenado con caras centradas y recortadas.

---

## 7. Mejoras propuestas (pendientes de probar)

### Preprocesamiento

- [ ] Activar rotación (`APLICAR_ROTACION = True`, ~15°)
- [ ] Activar recorte aleatorio (`APLICAR_RECORTE_ALEATORIO = True`)
- [ ] Probar resolución 64×64
- [ ] Comparar RGB vs escala de grises

### Entrenamiento

- [ ] Bajar learning rate a `5e-4`
- [ ] Subir dropout FC a `0.6`
- [ ] Agregar `weight_decay=1e-4` en Adam
- [ ] Probar `CrossEntropyLoss` con pesos por clase (complemento al sampler)
- [ ] Reducir paciencia de early stopping a 5

### Arquitectura

- [ ] Más capacidad (filtros o capa conv adicional)
- [ ] Global Average Pooling para reducir overfitting
- [ ] Dropout2d en capas convolucionales

### Inferencia (puntos 4 y 5)

- [ ] Afinar Haar: `SCALE_FACTOR=1.05`, `MIN_NEIGHBORS=5`, `MIN_SIZE=(40,40)`
- [ ] Documentar aciertos punto 4 vs punto 5 con las mismas imágenes

### Análisis cualitativo

- [ ] Revisar visualmente ejemplos mal clasificados tristeza/seriedad y disgusto/seriedad

---

## 8. Tabla comparativa de experimentos

| # | Cambio | Acc val | F1 macro | Mejor época | Notas |
|---|--------|---------|----------|-------------|-------|
| 1 | Baseline: balanceo + early stop + LR 1e-3, 48×48 gris | 73.3% | 65.8% | 19 | Resultado actual |
| 2 | | | | | |
| 3 | | | | | |

---

*Última actualización: junio 2026*
