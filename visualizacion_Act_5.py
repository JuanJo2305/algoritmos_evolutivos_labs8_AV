import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============ 1. Evolución del fitness por generación ============

# Leer evolución del fitness para cada representación
df_bin = pd.read_csv('fitness_binaria.csv')
df_real = pd.read_csv('fitness_real.csv')
df_perm = pd.read_csv('fitness_permutacional.csv')

df_bin['Representacion'] = 'Binaria'
df_real['Representacion'] = 'Real'
df_perm['Representacion'] = 'Permutacional'

df_fitness = pd.concat([df_bin, df_real, df_perm])

plt.figure(figsize=(10, 5))
sns.lineplot(data=df_fitness, x='Generacion', y='Fitness', hue='Representacion', marker='o')
plt.title('Evolución del Fitness por Generación')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend(title='Representación')
plt.tight_layout()
plt.savefig('fitness_evolucion_todas.png')
plt.show()

# ============ 2. Histograma de notas por examen (todas las representaciones) ============

# Leer asignaciones finales
df_asig_bin = pd.read_csv('asignaciones_binaria.csv')
df_asig_real = pd.read_csv('asignaciones_real.csv')
df_asig_perm = pd.read_csv('asignaciones_permutacional.csv')

df_asig_bin['Representacion'] = 'Binaria'
df_asig_real['Representacion'] = 'Real'
df_asig_perm['Representacion'] = 'Permutacional'

df_asignaciones = pd.concat([df_asig_bin, df_asig_real, df_asig_perm])

plt.figure(figsize=(12, 6))
sns.histplot(data=df_asignaciones, x='Nota', hue='Examen', multiple='stack', kde=True, bins=10, palette='Set2')
plt.title('Histograma de Notas por Examen (Todas las Representaciones)')
plt.xlabel('Nota')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('histograma_notas_todas.png')
plt.show()

# ============ 3. Comparar distribuciones de notas (violin plot) ============

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_asignaciones, x='Examen', y='Nota', hue='Representacion', palette='Pastel2', split=True)
plt.title('Distribución de Notas por Grupo y Representación')
plt.xlabel('Examen')
plt.ylabel('Nota')
plt.tight_layout()
plt.savefig('distribucion_notas_violin.png')
plt.show()

# ============ 4. Boxplot alternativo (opcional) ============

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_asignaciones, x='Examen', y='Nota', hue='Representacion', palette='Set3')
plt.title('Boxplot de Notas por Grupo y Representación')
plt.xlabel('Examen')
plt.ylabel('Nota')
plt.tight_layout()
plt.savefig('distribucion_notas_boxplot.png')
plt.show()
