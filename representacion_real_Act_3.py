import random
import numpy as np
import pandas as pd

# Para reproducibilidad
random.seed(42)
np.random.seed(42)

# Cargar datos
df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# 1. Crear cromosoma: 39 alumnos × 3 pesos normalizados
def crear_cromosoma():
    cromosoma = []
    for _ in range(39):
        pesos = [random.random() for _ in range(3)]
        total = sum(pesos)
        cromosoma.extend([p/total for p in pesos])
    return cromosoma

# 2. Decodificar: asignar cada alumno al examen de mayor peso
def decodificar_cromosoma(cromosoma):
    matriz = np.array(cromosoma).reshape((39, 3))
    asignaciones = {'A': [], 'B': [], 'C': []}
    for i, fila in enumerate(matriz):
        idx = int(np.argmax(fila))
        examen = ['A', 'B', 'C'][idx]
        asignaciones[examen].append(i)
    return asignaciones

# 3. Fitness: equilibrar promedios y varianza interna
def calcular_fitness(cromosoma):
    asign = decodificar_cromosoma(cromosoma)
    proms, vars_ = [], []
    for ex in ['A','B','C']:
        notas_ex = [notas[i] for i in asign[ex]]
        proms.append(np.mean(notas_ex))
        vars_.append(np.var(notas_ex))
    desv_prom = np.std(proms)
    mean_var = np.mean(vars_)
    # Fitness positivo mejor
    return 100 - 80*desv_prom - 10*mean_var

# 4. Cruce simple + gaussiano por alumno
def cruce(p1, p2):
    hijo = []
    for i in range(39):
        idx = i*3
        genes = p1[idx:idx+3] if random.random()<0.5 else p2[idx:idx+3]
        # añadimos ruido leve en el cruce
        hijo.extend(genes)
    return hijo

# 5. Mutación gaussiana manteniendo normalización
def mutacion_gaussiana(crom, sigma):
    crom_m = crom.copy()
    print(f"   [mutación gaussiana σ={sigma}]")
    for i in range(39):
        idx = i*3
        pesos = crom_m[idx:idx+3]
        nuevos = [max(0, p + random.gauss(0, sigma)) for p in pesos]
        s = sum(nuevos)
        if s>0:
            nuevos = [x/s for x in nuevos]
        else:
            nuevos = [1/3,1/3,1/3]
        crom_m[idx:idx+3] = nuevos
    return crom_m

# 6. Algoritmo genético
def algoritmo_genetico(generaciones, tam_pob, sigma):
    pobl = [crear_cromosoma() for _ in range(tam_pob)]
    historial = []
    print(f"→ Usando mutación gaussiana con σ = {sigma}")
    for gen in range(generaciones):
        scores = [(c, calcular_fitness(c)) for c in pobl]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_fit = scores[0][1]
        historial.append(best_fit)

        # Elitismo
        nueva = [scores[i][0] for i in range(int(0.1*tam_pob))]

        # Rellenar con cruce + mutación
        top = [c for c,_ in scores[:tam_pob//4]]
        while len(nueva)<tam_pob:
            p1 = random.choice(top)
            p2 = random.choice(top)
            h = cruce(p1,p2)
            h = mutacion_gaussiana(h, sigma)
            nueva.append(h)
        pobl = nueva

        if gen%30==0:
            print(f"Gen {gen:3d}: best fitness = {best_fit:.4f}")
    # mejor global
    return scores[0][0], historial

# 7. Probar distintos sigmas y comparar
def imprimir_tabla(res):
    print("\n╔════════════╦═════════════════╗")
    print("║   Sigma    ║  Mejor Fitness  ║")
    print("╠════════════╬═════════════════╣")
    for s,f in res:
        print(f"║  {s:<10}║  {f:<15.4f}║")
    print("╚════════════╩═════════════════╝")

print("\n=== PRUEBA MUTACIÓN GAUSSIANA ===")
resultados = []
for sigma in [0.01, 0.1, 0.5, 1.0, 2.0]:
    sol, hist = algoritmo_genetico(150, 100, sigma)
    fit_final = hist[-1]
    resultados.append((sigma, fit_final))
# mostrar comparación
imprimir_tabla(resultados)
# Decodificar y mostrar mejor caso (el último sigma)
asig = decodificar_cromosoma(sol)
print("\nDistribución final para σ =", resultados[-1][0])
for ex in ['A','B','C']:
    idxs = asig[ex]
    notas_ex = [notas[i] for i in idxs]
    print(f" Ex {ex}: {len(idxs)} alumnos, prom {np.mean(notas_ex):.2f}, var {np.var(notas_ex):.2f}")
