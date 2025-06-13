import random
import numpy as np
import pandas as pd

df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

def crear_cromosoma():
    asignaciones = [0]*13 + [1]*13 + [2]*13
    random.shuffle(asignaciones)
    
    cromosoma = []
    for examen in asignaciones:
        genes = [0, 0, 0]
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma


def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': []}
    examenes = ['A', 'B', 'C']
    
    for i in range(39):
        idx = i * 3
        for j in range(3):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    
    return asignaciones

def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    
    # Penalización si no hay 13 alumnos por grupo
    if any(len(asignaciones[ex]) != 13 for ex in ['A', 'B', 'C']):
        return -1000
    
    # Clasificación de notas: bajos, medios, altos
    notas_ordenadas = sorted(notas)
    tercio = len(notas) // 3
    umbral_bajo = notas_ordenadas[tercio]
    umbral_alto = notas_ordenadas[-tercio - 1]

    desviaciones_internas = []
    promedios = []
    diversidad_total = 0

    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_grupo = [notas[i] for i in indices]
        promedios.append(np.mean(notas_grupo))

        # 1. Varianza interna
        varianza = np.var(notas_grupo)
        desviaciones_internas.append(varianza)

        # 2. Diversidad
        categorias = set()
        for nota in notas_grupo:
            if nota <= umbral_bajo:
                categorias.add('bajo')
            elif nota >= umbral_alto:
                categorias.add('alto')
            else:
                categorias.add('medio')
        diversidad_total += len(categorias)  # ideal: 3 por grupo → 9 en total

    # 3. Desviación entre promedios
    desviacion_entre_grupos = np.std(promedios)

    # Penalizar desviación entre promedios y varianzas internas
    penalizacion = desviacion_entre_grupos + np.mean(desviaciones_internas)

    # Premiar diversidad: máximo posible = 9, mínimo = 3
    bonus_diversidad = diversidad_total / 9  # va de 0.33 a 1.0

    fitness = -penalizacion + bonus_diversidad  # maximizar fitness

    return fitness


def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()

    # Elegimos dos alumnos al azar
    alumno1 = random.randint(0, 38)
    alumno2 = random.randint(0, 38)

    idx1 = alumno1 * 3
    idx2 = alumno2 * 3

    examen1 = cromosoma_mutado[idx1:idx1+3].index(1)
    examen2 = cromosoma_mutado[idx2:idx2+3].index(1)

    # Intercambiar solo si son diferentes y mantener equilibrio
    if examen1 != examen2:
        cromosoma_mutado[idx1:idx1+3] = [0, 0, 0]
        cromosoma_mutado[idx1 + examen2] = 1

        cromosoma_mutado[idx2:idx2+3] = [0, 0, 0]
        cromosoma_mutado[idx2 + examen1] = 1

    return cromosoma_mutado




def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    
    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        nueva_poblacion = []
        
        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])
        
        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(poblacion[:tam_poblacion//2])
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        
        if gen % 20 == 0:
            mejor_fitness = fitness_scores[0][1]
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.4f}")
    
    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma

mejor_solucion = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución final con análisis de diversidad:")
notas_ordenadas = sorted(notas)
tercio = len(notas) // 3
umbral_bajo = notas_ordenadas[tercio]
umbral_alto = notas_ordenadas[-tercio - 1]

promedios = []
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    varianza = np.var(notas_examen)
    promedios.append(promedio)

    # Clasificación por categoría
    conteo = {'bajo': 0, 'medio': 0, 'alto': 0}
    for nota in notas_examen:
        if nota <= umbral_bajo:
            conteo['bajo'] += 1
        elif nota >= umbral_alto:
            conteo['alto'] += 1
        else:
            conteo['medio'] += 1

    print(f"\nExamen {examen}:")
    print(f"- {len(indices)} alumnos")
    print(f"- Promedio = {promedio:.2f}")
    print(f"- Varianza interna = {varianza:.4f}")
    print(f"- Diversidad: bajos = {conteo['bajo']}, medios = {conteo['medio']}, altos = {conteo['alto']}")
    print(f"  Ejemplos de alumnos: {[alumnos[i] for i in indices[:5]]}...")

print("\nVerificación de equilibrio:")
print(f"- Desviación estándar entre promedios: {np.std(promedios):.4f}")
