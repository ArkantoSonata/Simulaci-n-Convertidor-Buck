# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:59:32 2025

@author: 100199601
"""

import cmath
import numpy as np
import matplotlib.pyplot as plt

# --- PARÁMETROS DEL CONVERTIDOR ---
R = 15
L = 25e-6
C = 330e-6
Vin = 10.0
u = Vin
vref = 5.0
t_rampa = 1e-3  # Duración de la rampa en segundos (0.5 ms)


# --- EIGENVALORES Y VECTORES ---
f1 = 1 / (L * C)
f2 = 1 / (4 * (R * C)**2)
alpha = -1 / (2 * R * C)
beta = cmath.sqrt(f2 - f1)

lamb1_10 = alpha + beta
lamb2_10 = alpha - beta
lamb1_01 = alpha + beta
lamb2_01 = alpha - beta
lamb1_00 = 0
lamb2_00 = -1 / (C * R)

v1_10 = np.array([1, -lamb1_10 * L])
v2_10 = np.array([1, -lamb2_10 * L])
v1_01 = np.array([1, -lamb1_01 * L])
v2_01 = np.array([1, -lamb2_01 * L])
v1_00 = np.array([1, 0])
v2_00 = np.array([0, 1])



# --- CONTROLADOR DISCRETO PZM ---
class ControladorPZMatching:
    def __init__(self, K=0.0609, a=0.9851):
        self.K = K
        self.a = a
        self.d_prev = 0.5
        self.e_prev = 0.0

    def actualizar(self, v_ref, vC_actual):
        e = v_ref - vC_actual
        d = self.d_prev + self.K * (e - self.a * self.e_prev)
        d = max(0.0, min(0.5, d))
        self.d_prev = d
        self.e_prev = e
        return d

# --- FUNCIONES AUXILIARES ---
def calcular_constantes(x0, v1, v2, xp):
    A = np.column_stack((v1, v2))
    b = x0 - xp
    c = np.linalg.solve(A, b)
    return c[0], c[1]

# --- INTERPOLACIÓN DE NEWTON PARA DICM ---
def refinar_t_dicm_newton(t0, c1, c2, v1, v2, lamb1, lamb2, xp, il_umbral=1e-3, max_iter=10):
    """
    Refina el tiempo exacto en el que iL cruza por cero usando interpolación de Newton de segundo orden.
    """
    t = t0 + 1e-9
    for _ in range(max_iter):
        dt_local = t - t0
        exp1 = cmath.exp(lamb1 * dt_local)
        exp2 = cmath.exp(lamb2 * dt_local)

        iL = (c1 * v1[0] * exp1 + c2 * v2[0] * exp2 + xp[0]).real
        diL = (c1 * v1[0] * lamb1 * exp1 + c2 * v2[0] * lamb2 * exp2).real
        d2iL = (c1 * v1[0] * lamb1**2 * exp1 + c2 * v2[0] * lamb2**2 * exp2).real

        if abs(diL) < 1e-12 or abs(d2iL) < 1e-12:
            break

        dt = (1 / d2iL) * (0.5 / diL - iL / (diL**2))
        t_next = t + dt

        if abs(dt) < 1e-12:
            return t_next

        t = t_next

    return t  # Retorna mejor estimado

# --- SIMULACIÓN DEL ESTADO DINÁMICO ---
def resolver_estado(lamb1, lamb2, v1, v2, c1, c2, xp, t0, tf, dt=1e-7, il_umbral=1e-3, detectar_dicm=False):
    t_vals, x_vals = [], []
    t = t0
    prev_il = None

    while t <= tf:
        dt_local = t - t0
        exp1 = cmath.exp(lamb1 * dt_local)
        exp2 = cmath.exp(lamb2 * dt_local)
        xh = c1 * v1 * exp1 + c2 * v2 * exp2
        xt = xh + xp
        xt_real = xt.real
        
        # 🔧 FORZAR iL ≥ 0 en tiempo real
        xt_real[0] = max(xt_real[0], 0.0)
        
        il = xt_real[0]
        
        t_vals.append(t)
        x_vals.append(xt_real)


        if detectar_dicm and prev_il is not None and prev_il * il < 0:
            # Refinar el cruce por cero
            t_refinado = refinar_t_dicm_newton(t0, c1, c2, v1, v2, lamb1, lamb2, xp)
            dt_local_ref = t_refinado - t0
            exp1_ref = cmath.exp(lamb1 * dt_local_ref)
            exp2_ref = cmath.exp(lamb2 * dt_local_ref)
            xh_ref = c1 * v1 * exp1_ref + c2 * v2 * exp2_ref
            xt_ref = xh_ref + xp
            xt_real = xt_ref.real
            xt_real[0] = 0.0  # Forzar iL = 0

        
            # Agregar solo hasta el cruce
            t_vals.append(t_refinado)
            x_vals.append(xt_ref.real)
        
            print(f"🟥 DICM detectado en t = {t_refinado * 1e6:.2f} µs, iL = {xt_ref.real[0]:.4e}")

            return np.array(t_vals), np.array(x_vals)  # 🔁 Finalizar integración aquí


        prev_il = il
        t += dt

    return np.array(t_vals), np.array(x_vals)



# --- SIMULACIÓN ---
T_total = 7.5e-3
dt = 1e-8
Ts = 10e-6
controlador = ControladorPZMatching()
t = 0
x0 = np.array([0.0, 0.0])
t_all, x_all = [], []
duty_vals = []
t_duty_vals = []

en_dicm = False  # bandera para saber si estamos saliendo de DICM

tiempos_dicm = []

while t < T_total:
    # === 1. CONTROLADOR ===
    
    if t < t_rampa:
        tau = t / t_rampa  # tiempo normalizado entre 0 y 1
        vref_t = vref * tau**3  # rampa cúbica
    else:
        vref_t = vref


    vC_actual = x0[1]
    duty = controlador.actualizar(vref_t, vC_actual)

    # Duty mínimo adaptable (crece cuando estamos en DICM)
    duty_min = 0.05 if not en_dicm else 0.1
    duty = max(duty_min, min(0.95, duty))

    duty_vals.append(duty)
    t_duty_vals.append(t * 1e6)  # tiempo en µs

    # === 2. ESTADO "10" (switch ON) ===
    t_on = t
    t_off = t + duty * Ts
    xp = np.array([C / R, 1.0]) * u
    c1, c2 = calcular_constantes(x0, v1_10, v2_10, xp)
    t_vals, x_vals = resolver_estado(
        lamb1_10, lamb2_10, v1_10, v2_10, c1, c2, xp,
        t_on, t_off, dt, detectar_dicm=True
    )
    t_all.extend(t_vals)
    x_all.extend(x_vals)
    x0 = x_vals[-1]
    t = t_vals[-1]

    # === 3. VERIFICACIÓN DE DICM ===
    hubo_dicm = False
    if len(x_vals) >= 2 and x_vals[-2][0] * x_vals[-1][0] < 0:
        print(f"🟥 DICM detectado en t = {t*1e6:.2f} µs, iL = {x0[0]:.4e}")
        hubo_dicm = True
        en_dicm = True
        tiempos_dicm.append(t)  # Guardamos tiempo del cruce por cero


        # === 4. ESTADO "00" ===
        t_on = t
        t_off = t + 1e-6  # duración mínima
        xp = np.array([0.0, 0.0])
        c1, c2 = calcular_constantes(x0, v1_00, v2_00, xp)
        t_vals, x_vals = resolver_estado(
            lamb1_00, lamb2_00, v1_00, v2_00, c1, c2, xp,
            t_on, t_off, dt
        )
        print(f"⚫ Estado 00 desde {t_on*1e6:.2f} µs hasta {t_off*1e6:.2f} µs")
        t_all.extend(t_vals)
        x_all.extend(x_vals)
        x0 = x_vals[-1]
        t = t_vals[-1]

    # === 5. RECUPERACIÓN: forzar "10" si venimos de DICM ===
    if en_dicm:
        print(f"🔁 Saliendo de estado 00 → forzando switch ON en t = {t*1e6:.2f} µs")
        t_on = t
        t_off = t + max(1e-6, duty * Ts)
        xp = np.array([C / R, 1.0]) * u
        c1, c2 = calcular_constantes(x0, v1_10, v2_10, xp)
        t_vals, x_vals = resolver_estado(
            lamb1_10, lamb2_10, v1_10, v2_10, c1, c2, xp,
            t_on, t_off, dt
        )
        t_all.extend(t_vals)
        x_all.extend(x_vals)
        x0 = x_vals[-1]
        t = t_vals[-1]
        en_dicm = False  # se reinició el ciclo

    # === 6. ESTADO "01" (switch OFF, diodo ON) ===
    t_on = t
    t_off = t + (1 - duty) * Ts
    xp = np.array([C / R, 1.0]) * u
    c1, c2 = calcular_constantes(x0, v1_01, v2_01, xp)
    t_vals, x_vals = resolver_estado(
        lamb1_01, lamb2_01, v1_01, v2_01, c1, c2, xp,
        t_on, t_off, dt, detectar_dicm=True
    )
    t_all.extend(t_vals)
    x_all.extend(x_vals)
    x0 = x_vals[-1]
    t = t_vals[-1]
    
    # === 7. VERIFICACIÓN DE DICM TAMBIÉN EN MODO 01 ===
    hubo_dicm = False
    if len(x_vals) >= 2 and x_vals[-2][0] * x_vals[-1][0] < 0:
        print(f"🟥 DICM detectado en t = {t*1e6:.2f} µs, iL = {x0[0]:.4e}")
        hubo_dicm = True
        en_dicm = True
        tiempos_dicm.append(t)
    
        # === 8. ESTADO "00" ===
        t_on = t
        t_off = t + 1e-6  # duración mínima
        xp = np.array([0.0, 0.0])
        c1, c2 = calcular_constantes(x0, v1_00, v2_00, xp)
        t_vals, x_vals = resolver_estado(
            lamb1_00, lamb2_00, v1_00, v2_00, c1, c2, xp,
            t_on, t_off, dt
        )
        print(f"⚫ Estado 00 desde {t_on*1e6:.2f} µs hasta {t_off*1e6:.2f} µs")
        t_all.extend(t_vals)
        x_all.extend(x_vals)
        x0 = x_vals[-1]
        t = t_vals[-1]
    
        # === 9. RECUPERACIÓN: forzar "10" tras DICM ===
        print(f"🔁 Saliendo de estado 00 → forzando switch ON en t = {t*1e6:.2f} µs")
        t_on = t
        t_off = t + max(1e-6, duty * Ts)
        xp = np.array([C / R, 1.0]) * u
        c1, c2 = calcular_constantes(x0, v1_10, v2_10, xp)
        t_vals, x_vals = resolver_estado(
            lamb1_10, lamb2_10, v1_10, v2_10, c1, c2, xp,
            t_on, t_off, dt
        )
        t_all.extend(t_vals)
        x_all.extend(x_vals)
        x0 = x_vals[-1]
        t = t_vals[-1]
        en_dicm = False


# --- GRAFICAR iL(t) y vC(t) ---
t_all = np.array(t_all)
x_all = np.array(x_all)
iL = x_all[:, 0]
vC = x_all[:, 1]

# --- GRAFICAR iL(t) ---
plt.figure(figsize=(10, 4))
plt.plot(t_all * 1e6, iL, label="iL(t) [A]", color="blue")
for t_d in tiempos_dicm:
    plt.axvline(t_d * 1e6, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.text(t_d * 1e6, 0, 'DICM', color='red', rotation=90, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right')

plt.xlabel("Tiempo (µs)")
plt.ylabel("Corriente iL (A)")
plt.title("Corriente del inductor iL(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- GRAFICAR SOLO vC(t) ---
plt.figure(figsize=(10, 4))
plt.plot(t_all * 1e6, vC, label="vC(t) [V]", color="orange")

# Línea horizontal para vref
plt.axhline(y=vref, color='gray', linestyle='--', linewidth=1, alpha=0.7, label="vref")

# Marcar momento en que alcanza vref por primera vez (opcional)
idx_cruce = np.argmax(vC >= vref)
if vC[idx_cruce] >= vref:
    t_cruce = t_all[idx_cruce] * 1e6  # µs
    plt.axvline(x=t_cruce, color='purple', linestyle='--', linewidth=1)
    plt.text(t_cruce, vref + 0.1, f'Cruce vref ≈ {t_cruce:.1f} µs', color='purple', fontsize=8,
             ha='right', rotation=90)

plt.xlabel("Tiempo (µs)")
plt.ylabel("Voltaje del capacitor vC (V)")
plt.title("Evolución del voltaje vC(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- GRAFICAR DUTY CYCLE ---
plt.figure(figsize=(10, 4))
plt.step(t_duty_vals, duty_vals, where='post', color='green', label='Duty cycle', linewidth=2)

plt.xlabel("Tiempo (µs)")
plt.ylabel("Duty")
plt.title("Evolución del duty cycle (controlador PZ matching)")
plt.grid(True)
plt.legend()

# 🔍 ZOOM: Limitar el eje X a los primeros 500 µs (ajusta según necesites)
plt.xlim(0, 500)

# 🔍 Hacer más visible el duty (aunque sea pequeño)
plt.ylim(0, 0.5)

plt.tight_layout()
plt.show()

