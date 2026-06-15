# Conclusiones — TP3 Clasificador de Emociones

> **⚠️ REVISAR ANTES DE LA ENTREGA:** borrador actualizado con la config **E6** (modelo final). Verificar métricas, redacción y coherencia con el notebook (`§1.8`, `§3`) y [`reporte-experimentos.md`](reporte-experimentos.md) antes de enviar el Colab.

Documento de trabajo para el notebook `Luna-Marcelo-DL-TP3-CO24.ipynb`.

---

## 1. Resultados del modelo final (E6)

Configuración entregada: imágenes **64×64** en escala de grises, flip horizontal + ColorJitter, **sin** `WeightedRandomSampler`, **CrossEntropyLoss ponderada** (`sqrt_inverso`), early stopping en `val_f1`, dropout 0.5, Adam lr=1e-4.

| Métrica | Validación |
|---------|------------|
| Accuracy | **79.34%** |
| F1 macro | **71.30%** |
| Loss | **0.6097** |
| Mejor época | **29** (early stop en época **34**) |

### Rendimiento por clase (E6)

| Clase | Precision | Recall | F1 | Observación |
|-------|-----------|--------|-----|-------------|
| alegría | 0.91 | 0.90 | **0.90** | Mejor clase; estable en todas las configs |
| sorpresa | 0.81 | 0.79 | **0.80** | Buen rendimiento |
| seriedad | 0.76 | 0.78 | **0.77** | Mejor recall que en configs anteriores |
| tristeza | 0.70 | 0.71 | **0.71** | Sigue confundiéndose parcialmente con seriedad |
| enojo | 0.71 | 0.69 | **0.70** | Aceptable |
| miedo | 0.64 | 0.55 | **0.59** | Clase minoritaria; mejora leve vs E4 |
| disgusto | 0.51 | 0.53 | **0.52** | Sigue siendo la más difícil, pero mejor que E0–E4 |

El F1 macro (71.3%) queda más alineado con el accuracy (79.3%) que en el baseline (~66% vs ~75%), gracias a la compensación del desbalance vía loss ponderada (sin sampler).

---

## 2. Evolución del entrenamiento (experimentos E0–E6)

| Exp | Cambio principal | F1 macro | Resultado |
|-----|------------------|----------|-----------|
| E0 | Baseline (48×48, sampler, early stop `val_loss`) | 66.7% | Punto de partida |
| E1 | Early stopping → `val_f1` | 69.5% | Mejor criterio de checkpoint |
| E2 | Loss `sqrt_inverso` + sampler | 66.0% | Descartado (doble compensación) |
| E3 | Dropout FC 0.6 | 68.4% | Descartado |
| E4 | Resolución 64×64 | 69.7% | Mejor config hasta E6 |
| E5 | Rotación ±10° | 68.8% | Descartado |
| **E6** | Loss `sqrt_inverso` **sin** sampler | **71.3%** | **Config final** |

Aprendizajes clave:

- **Early stopping en `val_f1`** (E1) alinea el checkpoint con el objetivo de evaluación.
- **64×64** (E4) aporta detalle facial útil para pares confusos.
- **Loss ponderada sin sampler** (E6) supera sampler solo (E4) y sampler+loss (E2): un solo mecanismo de compensación es suficiente.
- Rotación suave (E5) y dropout extra (E3) no mejoraron sobre E4/E6.

---

## 3. Patrones en la matriz de confusión (E6)

1. **Tristeza ↔ seriedad:** sigue siendo el par más problemático, aunque E6 reduce confusiones respecto a configs con 48×48.
2. **Disgusto:** F1 ~0.52; recall ~0.53 con precision ~0.51 — más equilibrado que E2/E4, donde subía recall a costa de precision.
3. **Miedo ↔ sorpresa:** confusión residual por expresiones con ojos/boca abiertos; miedo F1 ~0.59.
4. **Train acc > val acc** (~87% vs ~79% en mejor época E6): muestreo natural (sin sampler) + augmentations solo en train; distinto al patrón con balanceo (E4).

Confusiones relevantes en validación (E6, matriz normalizada):

- Tristeza real → seriedad predicha: ~14% de tristeza.
- Disgusto real → seriedad predicha: ~14% de disgusto.
- Miedo real → sorpresa predicha: ~12% de miedo.

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

**Estrategia adoptada (E6):** pesos `sqrt_inverso` en la loss, **sin** oversampling en el DataLoader. Evita la sobre-compensación observada en E2 cuando se combinaron sampler y loss.

---

## 5. Inferencia en imágenes propias (puntos 4 y 5)

> **⚠️ REVISAR ANTES DE LA ENTREGA:** completar con resultados de fotos propias y ajustar redacción según lo observado.

| Entrenamiento / validación | Imágenes propias |
|----------------------------|------------------|
| Rostros alineados (`*_aligned.jpg`), 64×64 gris | Fotos completas, otro encuadre e iluminación |
| Pipeline `transform_val` (sin augmentations) | Recorte Haar (punto 5) recomendado |

Observaciones:

- Usar siempre **`transform_val`** en inferencia (64×64, sin rotación ni ColorJitter).
- El **punto 5** (Haar + recorte) suele ser más confiable que el punto 4 para fotos reales.
- Revisar scores softmax cuando top-1 y top-2 están muy cercanos.

Parámetros Haar finales (I3): `SCALE_FACTOR=1.05`, `MIN_NEIGHBORS=4`, `MIN_SIZE=(20,20)`, cascade default.

Resultados registrados (14 imágenes propias, modelo E6):
- **Punto 4** (sin Haar): 1/14 aciertos.
- **Punto 5** (Haar I3 + `transform_val`): **14/14** detección, **4/14** aciertos (`sorpresa 2`, `enojo 2`, `alegria 2`, `tristeza 2`).

---

## 6. Conclusiones generales (borrador para el informe / Colab)

> **⚠️ REVISAR ANTES DE LA ENTREGA:** adaptar este texto al formato de respuestas del TP y copiar/refinar en el notebook si corresponde.

- Se construyó una CNN desde cero (3 bloques conv 32-64-128, FC 256) con pipeline configurable en PyTorch.
- El preprocesamiento final usa **64×64 en escala de grises**, normalización, volteo horizontal y ColorJitter en entrenamiento.
- La **campaña experimental** (E0–E6) permitió aislar decisiones: early stopping en F1 macro, resolución 64×64 y loss ponderada sin sampler.
- **Mejor resultado:** E6 — **79.3% accuracy**, **71.3% F1 macro** en validación.
- Las clases más difíciles (**disgusto**, **miedo**) mejoran con E6 respecto al baseline, pero siguen limitando el F1 macro.
- Confusiones entre emociones visualmente parecidas (tristeza/seriedad, miedo/sorpresa) persisten; parte del techo es semántico y de resolución, no solo de hiperparámetros.
- En imágenes externas al dataset, el **alineamiento del rostro** (punto 5) y el uso de `transform_val` son críticos.

---

## 7. Pendientes pre-entrega

> **⚠️ REVISAR ANTES DE LA ENTREGA**

### Notebook / Colab

- [ ] Re-ejecutar §1.2, §2 y §5.1 y confirmar mensajes `✓ config E6 OK` / `✓ inferencia I3 + modelo E6 OK`
- [ ] Re-ejecutar §1.8 y verificar que refleja E6
- [ ] Confirmar outputs de §3 (E6) visibles en el Colab
- [ ] **Completar §3.6** — síntesis final desde `reporte-experimentos.md` (celda placeholder marcada ⬜ PENDIENTE)
- [x] **E6-b** — inferencia §4–§5 con conjunto alternativo (5/14 P5 vs 4/14 Set A; ver reporte § E6-b)
- [ ] Habilitar comentarios y acceso al correo de la cátedra

### Opcional (no necesario si E6 se congela)

- [x] Resolución 64×64 (E4)
- [x] Loss ponderada sin sampler (E6)
- [x] Rotación ±10° (E5 — descartada)
- [x] RGB vs gris (E7 — descartado; mantener E6 gris)
- [x] RandomCrop domain shift (E8 — descartado; −10.4 pp F1 vs E6)
- [x] Afinar Haar en inferencia (I0–I3; config final I3: scaleFactor 1.05)

---

## 8. Tabla comparativa de experimentos

| Exp | Cambio | Acc val | F1 macro | Mejor época | Notas |
|-----|--------|---------|----------|-------------|-------|
| E0 | Baseline | 75.4% | 66.7% | 17 | Ver reporte |
| E1 | Early stop `val_f1` | 76.8% | 69.5% | 16 | |
| E4 | 64×64 | 77.5% | 69.7% | 17 | Base previa a E6 |
| E5 | Rotación ±10° | 77.0% | 68.8% | 32 | Descartado |
| **E6** | Loss sin sampler | **79.3%** | **71.3%** | **29** | **Final** |
| E7 | RGB | 79.0% | 70.9% | 25 | Descartado |
| E8 | RandomCrop train | 70.1% | 60.9% | 24 | Descartado |

Detalle completo: [`reporte-experimentos.md`](reporte-experimentos.md).

---

*Última actualización: 2026-06-19 — E6 config final; E7–E8 descartados. Campaña reentrenamiento cerrada. ⚠️ Revisar antes de entregar.*
