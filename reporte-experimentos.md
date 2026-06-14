# Reporte de experimentos — TP3 Clasificador de Emociones

Registro de corridas a partir de **2026-06-06**.  
Notebook: `Luna-Marcelo-DL-TP3-CO24.ipynb`

---

## Cómo usar este documento

1. **Antes de entrenar:** anotar el ID del experimento (E0, E1, …) y copiar la config activa del notebook.
2. **Al cambiar algo:** registrar solo lo que difiere respecto al experimento anterior (preprocesamiento, modelo o entrenamiento).
3. **Después de entrenar:** completar métricas de validación y observaciones breves.
4. **Actualizar** la tabla comparativa al final de cada corrida.

Campos prioritarios por experimento:

| Categoría | Qué registrar |
|-----------|---------------|
| **Preprocesamiento** | `TAMANO_IMAGEN`, `MODO_COLOR`, augmentations (`APLICAR_*`), balanceo |
| **Modelo / entrenamiento** | arquitectura (`FILTROS_CONV`, `DROPOUT_FC`, …), `LEARNING_RATE`, `OPTIMIZADOR`, early stopping |
| **Métricas** | accuracy, F1 macro, loss val, mejor época, época de parada |

---

## E0 — Baseline (config actual del notebook)

> Punto de partida para comparar experimentos.  
> **Corrida completada** — métricas de la mejor época restaurada (17).

### Preprocesamiento

| Parámetro | Valor |
|-----------|-------|
| `TAMANO_IMAGEN` | `(48, 48)` |
| `MODO_COLOR` | `"grayscale"` → 1 canal |
| `NORMALIZE_MEAN` / `STD` | `[0.5]` / `[0.5]` |
| `APLICAR_VOLTEO_HORIZONTAL` | `True` |
| `APLICAR_ROTACION` | `False` (`ROTACION_GRADOS = 15`) |
| `APLICAR_RECORTE_ALEATORIO` | `False` (`RECORTE_TAMANO = (44, 44)`) |
| `APLICAR_COLOR_JITTER` | `True` (brightness `0.2`, contrast `0.2`) |
| `APLICAR_BALANCEO` | `True` (`WeightedRandomSampler`) |

### Modelo

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | 3 bloques Conv2d→BN→ReLU→MaxPool |
| `FILTROS_CONV` | `[32, 64, 128]` |
| `TAMANO_KERNEL` / `PADDING_CONV` | `3` / `1` |
| `TAMANO_POOL` | `2` |
| `UNIDADES_FC_OCULTAS` | `256` |
| `DROPOUT_FC` | `0.5` |
| Loss | `CrossEntropyLoss` |

### Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| `BATCH_SIZE` | `32` |
| `NUM_EPOCHS` | `40` (máx.) |
| `LEARNING_RATE` | `1e-4` |
| `OPTIMIZADOR` | `"adam"` |
| `APLICAR_EARLY_STOPPING` | `True` |
| `EARLY_STOPPING_METRICA` | `"val_loss"` |
| `EARLY_STOPPING_PATIENCE` | `5` |
| `EARLY_STOPPING_MIN_DELTA` | `0.0` |
| `EARLY_STOPPING_RESTAURAR_MEJOR_MODELO` | `True` |

### Inferencia punto 5 (Haar)

| Parámetro | Valor |
|-----------|-------|
| `SCALE_FACTOR` | `1.1` |
| `MIN_NEIGHBORS` | `6` |
| `MIN_SIZE` | `(30, 30)` |

### Métricas — validación

| Métrica | Valor |
|---------|-------|
| Accuracy | **75.42%** (0.7542) |
| F1 macro | **66.71%** (0.6671) |
| Loss | **0.6924** |
| Mejor época | **17** (`val_loss` = 0.6924) |
| Época de parada | **22** (early stopping, patience 5) |

#### F1 por clase (validación)

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| alegría | 0.93 | 0.84 | **0.88** |
| sorpresa | 0.79 | 0.79 | **0.79** |
| seriedad | 0.69 | 0.76 | 0.72 |
| enojo | 0.63 | 0.71 | 0.67 |
| tristeza | 0.65 | 0.68 | 0.66 |
| miedo | 0.55 | 0.50 | **0.52** |
| disgusto | 0.41 | 0.44 | **0.43** |

### Observaciones

- Early stopping restauró pesos de la **época 17**; a partir de ahí el train siguió bajando loss mientras val empeoraba → overfitting moderado.
- Clases más débiles: **disgusto** (F1 0.43) y **miedo** (F1 0.52); alegría concentra la mayoría de muestras y alcanza el mejor F1 (0.88).
- F1 macro (66.7%) por debajo del accuracy (75.4%): las clases minoritarias o difíciles arrastran el rendimiento global.

---

## Tabla comparativa

Completar una fila por corrida. En **Cambio vs anterior** resumir en una línea qué se modificó.

| Exp | Fecha | Cambio vs anterior | Acc val | F1 macro | Loss val | Mejor época | Parada | Notas |
|-----|-------|--------------------|---------|----------|----------|-------------|--------|-------|
| E0 | 2026-06-06 | Baseline (config actual) | 75.4% | 66.7% | 0.69 | 17 | 22 | Disgusto/miedo débiles |
| E1 | 2026-06-14 | Early stopping → `val_f1` | 76.8% | 69.5% | 0.68 | 16 | 21 | +2.8 pp F1 vs E0; disgusto ↑ |
| E2 | 2026-06-15 | Loss ponderada `sqrt_inverso` | 73.9% | 66.0% | 0.77 | 19 | 24 | Peor que E1; doble compensación |
| E3 | 2026-06-16 | Dropout FC 0.5 → 0.6 | 76.4% | 68.4% | 0.69 | 18 | 23 | Ligeramente peor que E1 |
| E4 | 2026-06-17 | Resolución 48×48 → 64×64 | 77.5% | 69.7% | 0.65 | 17 | 22 | Mejor F1 global; +0.2 pp vs E1 |
| E5 | 2026-06-18 | RandomRotation ±10° | 77.0% | 68.8% | 0.64 | 32 | 37 | Peor que E4; mantener E4 |
| E6 | 2026-06-19 | Loss `sqrt_inverso` sin sampler | 79.3% | 71.3% | 0.61 | 29 | 34 | **Mejor global**; +1.6 pp F1 vs E4 |

---

## Registro cronológico

### E0 — Baseline

**Fecha:** 2026-06-06  
**Estado:** corrida completada

**Preprocesamiento:** 48×48, gris, normalización 0.5/0.5, flip horizontal + ColorJitter (0.2/0.2), balanceo activo. Sin rotación ni recorte aleatorio.

**Modelo / entrenamiento:** CNN 32-64-128, FC 256, dropout 0.5, Adam lr=1e-4, batch 32, early stopping en val_loss (patience 5).

**Métricas — validación (mejor época 17):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 75.42% | 66.71% | 0.6924 | 17 | 22 |

**Observaciones:**
- Overfitting a partir de época 17 (train acc ~88% vs val ~75% en época 22).
- Peores clases: disgusto (F1 0.43), miedo (F1 0.52).
- Mejor clase: alegría (F1 0.88).

---

### E1 — Early stopping en `val_f1`

**Fecha:** 2026-06-14  
**Estado:** corrida completada

**Cambio vs E0:**
- Preprocesamiento: ninguno (misma config E0)
- Modelo: ninguno
- Entrenamiento: `EARLY_STOPPING_METRICA = "val_f1"` (antes `"val_loss"`); resto igual (patience 5, Adam lr=1e-4)

**Métricas — validación (mejor época 16, criterio val_f1 = 0.6945):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 76.79% | 69.45% | 0.6752 | 16 | 21 |

#### F1 por clase vs E0

| Clase | F1 E0 | F1 E1 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.88 | — |
| sorpresa | 0.79 | 0.80 | +0.01 |
| seriedad | 0.72 | 0.75 | +0.03 |
| enojo | 0.67 | 0.69 | +0.02 |
| tristeza | 0.66 | 0.66 | — |
| miedo | 0.52 | 0.58 | +0.06 |
| disgusto | 0.43 | 0.51 | **+0.08** |

**Observaciones:**
- Mejora global respecto a E0: **+1.4 pp accuracy**, **+2.7 pp F1 macro**, loss val ligeramente menor.
- El criterio `val_f1` priorizó un checkpoint (ép. 16) con mejor balance entre clases; disgusto y miedo suben notablemente.
- Disgusto: recall 0.55 (E0: 0.44), acierto diagonal en matriz norm. ~55% (E0: ~44%).
- Miedo: sube F1 pero recall baja (0.47 vs 0.50); gana precision (0.76 vs 0.55) — más conservador al predecir miedo.

---

### E2 — Loss ponderada por clase (`sqrt_inverso`)

**Fecha:** 2026-06-15  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: ninguno (misma config E0/E1)
- Modelo: `CrossEntropyLoss(weight=...)` con `APLICAR_PESOS_EN_LOSS = True`, `MODO_PESOS_CLASE = "sqrt_inverso"`
- Entrenamiento: early stopping en `val_f1` (igual E1); balanceo activo (igual E1)

**Pesos de loss (sqrt_inverso, media normalizada = 1):**

| Clase | n train | Peso |
|-------|---------|------|
| alegria | 4772 | 0.46 |
| disgusto | 717 | 1.19 |
| enojo | 705 | 1.20 |
| miedo | 281 | 1.90 |
| seriedad | 2524 | 0.64 |
| sorpresa | 1290 | 0.89 |
| tristeza | 1982 | 0.72 |

**Métricas — validación (mejor época 19, criterio val_f1 = 0.6601):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 73.92% | 66.01% | 0.7725 | 19 | 24 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E2 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.87 | −0.01 |
| sorpresa | 0.80 | 0.77 | −0.03 |
| seriedad | 0.75 | 0.72 | −0.03 |
| enojo | 0.69 | 0.66 | −0.03 |
| tristeza | 0.66 | 0.60 | −0.06 |
| miedo | 0.58 | 0.56 | −0.02 |
| disgusto | 0.51 | 0.44 | **−0.07** |

**Observaciones:**
- **Regresión global vs E1:** −2.9 pp accuracy, −3.4 pp F1 macro. El checkpoint elegido por `val_f1` quedó por debajo del de E1.
- **Doble compensación:** sampler (`WeightedRandomSampler`) + loss ponderada parecen sobre-corregir el desbalance; la loss escalada (val ~0.77 vs ~0.68 en E1) no es directamente comparable, pero las métricas de clasificación sí empeoran.
- **Disgusto:** recall sube (0.59 vs 0.55 en E1) pero precision cae fuerte (0.35); más falsos positivos → F1 baja a 0.44. Confusiones hacia disgusto desde seriedad (65/680) y tristeza (44/478).
- **Miedo:** precision alta (0.75) pero recall bajo (0.45); patrón similar a E1 pero peor F1 global.
- **Conclusión preliminar:** mantener E1 como base; si se reintenta loss ponderada, probar **sin sampler** o con `MODO_PESOS_CLASE = "balanceado"` solo en ablation — no acumular ambos mecanismos sin control.

---

### E3 — Dropout FC 0.6 (regularización)

**Fecha:** 2026-06-16  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: ninguno
- Modelo: `DROPOUT_FC = 0.6` (E1: 0.5)
- Entrenamiento: early stopping en `val_f1`, balanceo activo, `CrossEntropyLoss` sin pesos (igual E1)

**Métricas — validación (mejor época 18, criterio val_f1 = 0.6837):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 76.43% | 68.37% | 0.6913 | 18 | 23 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E3 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.89 | +0.01 |
| tristeza | 0.66 | 0.68 | +0.02 |
| enojo | 0.69 | 0.69 | — |
| disgusto | 0.51 | 0.50 | −0.01 |
| seriedad | 0.75 | 0.72 | −0.03 |
| sorpresa | 0.80 | 0.77 | −0.03 |
| miedo | 0.58 | 0.54 | −0.04 |

**Observaciones:**
- **No supera E1:** −0.4 pp accuracy, −1.1 pp F1 macro vs E1 (69.5%). El dropout extra no compensó la pérdida de capacidad útil en la FC.
- **Brecha train/val algo menor** en la mejor época (train acc ~82% vs val ~76%, gap ~6 pp) frente al overfitting más marcado de E1 — la regularización actuó, pero el checkpoint óptimo en `val_f1` quedó por debajo.
- **Miedo y sorpresa** bajan; **tristeza** sube levemente (+0.02). Disgusto prácticamente igual (0.50 vs 0.51).
- **Conclusión:** mantener **E1** (`DROPOUT_FC = 0.5`) como base. Próximo candidato: **E4** (64×64) o **E3b** (`weight_decay` sin subir dropout).

---

### E4 — Resolución 64×64

**Fecha:** 2026-06-17  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: `TAMANO_IMAGEN = (64, 64)` (E1: 48×48); resto igual (gris, flip + ColorJitter, balanceo)
- Modelo: misma arquitectura; FC se recalcula automáticamente (8192 → 256 vs 1152 → 256 en 48×48)
- Entrenamiento: early stopping `val_f1`, dropout 0.5, loss sin pesos (igual E1)

**Métricas — validación (mejor época 17, criterio val_f1 = 0.6969):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 77.48% | 69.69% | 0.6464 | 17 | 22 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E4 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.89 | +0.01 |
| seriedad | 0.75 | 0.75 | — |
| miedo | 0.58 | 0.58 | — |
| enojo | 0.69 | 0.70 | +0.01 |
| tristeza | 0.66 | 0.67 | +0.01 |
| sorpresa | 0.80 | 0.79 | −0.01 |
| disgusto | 0.51 | 0.49 | −0.02 |

**Observaciones:**
- **Mejor resultado global hasta ahora:** +0.7 pp accuracy, +0.2 pp F1 macro vs E1; loss val más baja (0.65 vs 0.68).
- La mejora es **modesta**; disgusto sigue débil (0.49) e incluso baja levemente vs E1. Miedo se mantiene en 0.58.
- **Seriedad** mejora recall (0.82 vs ~0.76 en E1) con precision estable — menos confusiones desde tristeza/disgusto hacia seriedad en matriz norm.
- **Miedo:** recall 0.53, precision 0.65; diagonal ~53% (similar a E1).
- **Conclusión:** **E4** fue la mejor config hasta E6. Superseded por E6 (+1.6 pp F1 macro).

---

### E5 — Rotación aleatoria ±10°

**Fecha:** 2026-06-18  
**Estado:** corrida completada

**Cambio vs E4:**
- Preprocesamiento: `APLICAR_ROTACION = True`, `ROTACION_GRADOS = 10` (solo train); resto igual (64×64, flip, ColorJitter, balanceo)
- Modelo / entrenamiento: sin cambios (early stopping `val_f1`, dropout 0.5, loss sin pesos)

**Métricas — validación (mejor época 32, criterio val_f1 = 0.6875):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 77.02% | 68.75% | 0.6445 | 32 | 37 |

#### F1 por clase vs E4

| Clase | F1 E4 | F1 E5 | Δ |
|-------|-------|-------|---|
| alegría | 0.89 | 0.89 | — |
| tristeza | 0.67 | 0.70 | +0.03 |
| enojo | 0.70 | 0.68 | −0.02 |
| seriedad | 0.75 | 0.73 | −0.02 |
| sorpresa | 0.79 | 0.78 | −0.01 |
| disgusto | 0.49 | 0.49 | — |
| miedo | 0.58 | 0.53 | **−0.05** |

**Observaciones:**
- **No supera E4:** −0.5 pp accuracy, −0.9 pp F1 macro vs E4 (69.7%). Entrenamiento más largo (mejor época 32, parada 37) sin ganancia en el checkpoint óptimo.
- **Disgusto:** F1 igual (0.49); sube recall (0.54 vs 0.46 en E4) pero baja precision (0.45 vs 0.53) — más falsos positivos.
- **Miedo** empeora (−0.05 F1); posible efecto de rotación en expresiones con ojos/boca abiertos.
- **Tristeza** mejora levemente (+0.03); no compensa el retroceso global.
- **Conclusión:** revertir a **E4** (rotación desactivada). E6 probado → supera E4.

---

### E6 — Loss `sqrt_inverso` sin sampler

**Fecha:** 2026-06-19  
**Estado:** corrida completada

**Cambio vs E4:**
- Preprocesamiento: 64×64, flip + ColorJitter (igual E4); **`APLICAR_BALANCEO = False`** (E4: sampler activo)
- Modelo: **`CrossEntropyLoss(weight=...)`, `MODO_PESOS_CLASE = "sqrt_inverso"`** (E4: sin pesos)
- Entrenamiento: early stopping `val_f1`, dropout 0.5 (igual E4)

**Métricas — validación (mejor época 29, criterio val_f1 = 0.7130):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 79.34% | 71.30% | 0.6097 | 29 | 34 |

#### F1 por clase vs E4

| Clase | F1 E4 | F1 E6 | Δ |
|-------|-------|-------|---|
| alegría | 0.89 | 0.90 | +0.01 |
| seriedad | 0.75 | 0.77 | +0.02 |
| sorpresa | 0.79 | 0.80 | +0.01 |
| tristeza | 0.67 | 0.71 | **+0.04** |
| enojo | 0.70 | 0.70 | — |
| miedo | 0.58 | 0.59 | +0.01 |
| disgusto | 0.49 | 0.52 | **+0.03** |

#### Comparativa de estrategias de desbalance

| Exp | Sampler | Loss ponderada | F1 macro |
|-----|---------|----------------|----------|
| E4 | sí | no | 69.7% |
| E2 | sí | sí | 66.0% |
| **E6** | **no** | **sí** | **71.3%** |

**Observaciones:**
- **Mejor resultado global de la campaña:** +1.9 pp accuracy, +1.6 pp F1 macro vs E4; +4.6 pp F1 vs E0.
- La ablation confirma la hipótesis de E2: el problema fue la **doble compensación**, no la loss ponderada en sí.
- **Disgusto** sube a F1 0.52 (recall 0.53, precision 0.51) — mejor equilibrio que E4/E2.
- **Tristeza** +0.04 F1; **seriedad** +0.02; todas las clases ≥ 0.52 F1.
- Train acc ~87% vs val ~79% en mejor época — overfitting moderado pero checkpoint óptimo claro.
- **Conclusión:** **E6 pasa a ser la config final recomendada** (64×64 + loss sqrt_inverso, sin sampler).

---

## Plantilla para nuevos experimentos

Copiar, renumerar (E1, E2, …) y completar:

```markdown
### EN — [Título breve]

**Fecha:** YYYY-MM-DD  
**Estado:** corrida completada | pendiente

**Cambio vs E(n-1):**
- Preprocesamiento: (ninguno | detalle)
- Modelo: (ninguno | detalle)
- Entrenamiento: (ninguno | detalle)

**Config completa** *(solo si cambió algo; si no, referenciar experimento base)*

| Preprocesamiento | Modelo | Entrenamiento |
|------------------|--------|---------------|
| … | … | … |

**Métricas — validación**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| | | | | |

**Observaciones:**
- 
```

---

*Última actualización: 2026-06-19 — E6 registrado (loss sin sampler; **mejor F1 macro 71.3%**). Config final: E6.*
