import random
import numpy as np
import pandas as pd

# Cargar datos
df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# Crear cromosoma (4 bits por alumno)
def crear_cromosoma():
    cromosoma = []
    for i in range(39):
        examen = random.randint(0, 3)  # 0=A, 1=B, 2=C, 3=D
        genes = [0, 0, 0, 0]
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma

# Decodificar cromosoma a asignaciones por examen
def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': [], 'D': []}
    examenes = ['A', 'B', 'C', 'D']
    
    for i in range(39):
        idx = i * 4
        for j in range(4):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    return asignaciones

# Evaluar fitness: buscamos mínima desviación estándar entre promedios
def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    tamaños = [len(asignaciones[ex]) for ex in ['A', 'B', 'C', 'D']]
    
    # Penaliza si no hay equilibrio (deben ser 9 o 10 alumnos por grupo)
    if any(t < 9 or t > 10 for t in tamaños):
        return -1000
    
    promedios = []
    for examen in ['A', 'B', 'C', 'D']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios.append(np.mean(notas_examen))
    
    desviacion = np.std(promedios)
    return -desviacion  # Queremos minimizar la desviación

# Mutación por intercambio de exámenes entre 2 alumnos
def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    
    alumno1 = random.randint(0, 38)
    alumno2 = random.randint(0, 38)
    
    idx1 = alumno1 * 4
    idx2 = alumno2 * 4
    
    ex1 = [i for i in range(4) if cromosoma_mutado[idx1 + i] == 1][0]
    ex2 = [i for i in range(4) if cromosoma_mutado[idx2 + i] == 1][0]
    
    if ex1 != ex2:
        cromosoma_mutado[idx1:idx1+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx1 + ex2] = 1
        
        cromosoma_mutado[idx2:idx2+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx2 + ex1] = 1
    
    return cromosoma_mutado

# Algoritmo Genético principal
def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    historial = []
    
    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        mejor_fitness = fitness_scores[0][1]
        historial.append(mejor_fitness)

        nueva_poblacion = []

        # Elitismo
        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])
        
        # Rellenar con mutaciones
        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(poblacion[:tam_poblacion//2])
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        
        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.4f}")
    
    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma, historial

# Ejecutar
print("REPRESENTACIÓN BINARIA")
print("Problema: Distribuir 39 alumnos en 4 exámenes (A, B, C, D) de forma casi equitativa")
print("Cromosoma: 156 bits (39 alumnos × 4 bits cada uno)\n")

mejor_solucion, historial = algoritmo_genetico()

# Mostrar resultados finales
asignaciones_finales = decodificar_cromosoma(mejor_solucion)
print("\nDistribución final:")
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Alumnos: {[alumnos[i] for i in indices[:5]]}... (mostrando primeros 5)")

print("\nVerificación de equilibrio:")
promedios = []
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")
