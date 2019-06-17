import torch
import sys, traceback
import numpy as np


def Heaviside(t):
    return (t > 0).float()


def cap_exp(t):
    return torch.exp(torch.clamp(t, max=14))


def func_init_deriv_0x0x1(t, a, b, c, c1, c2, sigma, epsilon):
    fc1 = torch.zeros_like(t)
    fc2 = torch.zeros_like(t)
    return fc1, fc2


def func_init_deriv_0x1x0(t, a, b, c, c1, c2, sigma, epsilon):
    fc1 = torch.ones_like(t)
    fc2 = torch.zeros_like(t)
    return fc1, fc2


def func_init_deriv_0x1x1(t, a, b, c, c1, c2, sigma, epsilon): 
    t1 = 0.1e1 / b
    t2 = t1 * c
    t4 = cap_exp(-t * t2)
    fc1 = t4
    fc2 = torch.zeros_like(t)
    return fc1, fc2


def func_init_deriv_1x0x0(t, a, b, c, c1, c2, sigma, epsilon):
    fc1 = t
    fc2 = torch.ones_like(t)
    return fc1, fc2


def func_init_deriv_1x0x1_a_neg_c_neg(t, a, b, c, c1, c2, sigma, epsilon):
    t2 = torch.sqrt(c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = torch.sin(t5)
    t8 = torch.cos(t5)
    fc1 = t8
    fc2 = t6
    return fc1, fc2


def func_init_deriv_1x0x1_a_neg_c_pos(t, a, b, c, c1, c2, sigma, epsilon):
    t2 = torch.sqrt(-c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = cap_exp(t5)
    t8 = cap_exp(-t5)
    fc1 = t8
    fc2 = t6
    return fc1, fc2


def func_init_deriv_1x0x1_a_pos_c_neg(t, a, b, c, c1, c2, sigma, epsilon):
    t2 = torch.sqrt(-c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = cap_exp(t5)
    t8 = cap_exp(-t5)
    fc1 = t8
    fc2 = t6
    return fc1, fc2


def func_init_deriv_1x0x1_a_pos_c_pos(t, a, b, c, c1, c2, sigma, epsilon):
    t2 = torch.sqrt(c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = torch.sin(t5)
    t8 = torch.cos(t5)
    fc1 = t8
    fc2 = t6
    return fc1, fc2


def func_init_deriv_1x1x0(t, a, b, c, c1, c2, sigma, epsilon):
    t3 = 0.1e1 / a
    t6 = cap_exp(-t * t3 * b)
    t13 = 1 / b
    t14 = t6 * t13
    fc1 = -a * t14
    fc2 = torch.ones_like(t)
    return fc1, fc2


def func_init_deriv_1x1x1_Delta_neg(t, a, b, c, c1, c2, sigma, epsilon):
    t1 = 0.1e1 / a
    t2 = t1 * b
    t5 = cap_exp(-t * t2 / 2)
    t8 = b ** 2
    t9 = 4 * a * c - t8
    t10 = torch.sqrt(t9)
    t13 = t * t1 * t10 / 2
    t14 = torch.sin(t13)
    t15 = t14 * t5
    t17 = torch.cos(t13)
    t18 = t17 * t5
    fc1 = t18
    fc2 = t15
    return fc1, fc2


def func_init_deriv_1x1x1_Delta_eq_0(t, a, b, c, c1, c2, sigma, epsilon):
    t1 = 1 / b
    t2 = c * t1
    t5 = cap_exp(-2 * t * t2)
    t6 = c2 * t5
    t7 = t * t5
    fc1 = t7
    fc2 = t5
    return fc1, fc2


def func_init_deriv_1x1x1_Delta_pos(t, a, b, c, c1, c2, sigma, epsilon):
    t3 = b ** 2
    t4 = -4 * a * c + t3
    t5 = torch.sqrt(t4)
    t6 = -b + t5
    t7 = 1 / a
    t8 = t7 * t6
    t11 = cap_exp(t * t8 / 2)
    t12 = c2 * t11
    t13 = b + t5
    t14 = t * t13
    t17 = cap_exp(-t7 * t14 / 2)
    fc1 = t17
    fc2 = t11
    return fc1, fc2


def func_init_deriv_1x0x1(t, a, b, c, c1, c2, sigma, epsilon):
    if a < 0 and c < 0:
        return func_init_deriv_1x0x1_a_neg_c_neg(t, a, b, c, c1, c2, sigma, epsilon)
    if a < 0 and c > 0:
        return func_init_deriv_1x0x1_a_neg_c_pos(t, a, b, c, c1, c2, sigma, epsilon)
    if a > 0 and c < 0:
        return func_init_deriv_1x0x1_a_pos_c_neg(t, a, b, c, c1, c2, sigma, epsilon)
    if a > 0 and c > 0:
        return func_init_deriv_1x0x1_a_pos_c_pos(t, a, b, c, c1, c2, sigma, epsilon)
    

def func_init_deriv_1x1x1(t, a, b, c, c1, c2, sigma, epsilon):
    Delta = b ** 2 - 4 * a * c
    if torch.abs(Delta) < epsilon:
        #b = torch.sign(b) * torch.sqrt(4 * a * c)
        a_sign = torch.sign(a)
        b_abs = torch.abs(b)
        c_sign = torch.sign(c)
        c = b_abs * c_sign / 2.
        a = b_abs * a_sign / 2.

        return func_init_deriv_1x1x1_Delta_eq_0(t, a, b, c, c1, c2, sigma, epsilon)
      
    if Delta < 0:
        return func_init_deriv_1x1x1_Delta_neg(t, a, b, c, c1, c2, sigma, epsilon)
    if Delta > 0:
        return func_init_deriv_1x1x1_Delta_pos(t, a, b, c, c1, c2, sigma, epsilon)

def deu_horse_init_deriv(t, a, b, c, c1, c2, fprime=False, sigma=100, epsilon=.01, max_val=1000.):
    
        
    a_abs = torch.abs(a)
    b_abs = torch.abs(b)
    c_abs = torch.abs(c)
    
    if a_abs < epsilon and b_abs < epsilon and c_abs < epsilon:
        ret = [torch.zeros_like(t), torch.zeros_like(t)]
    elif a_abs < epsilon and b_abs < epsilon and c_abs >= epsilon:
        ret = func_init_deriv_0x0x1(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs < epsilon and b_abs >= epsilon and c_abs < epsilon:
        ret = func_init_deriv_0x1x0(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs < epsilon and b_abs >= epsilon and c_abs >= epsilon:
        ret = func_init_deriv_0x1x1(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs >= epsilon and b_abs < epsilon and c_abs < epsilon:
        ret = func_init_deriv_1x0x0(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs >= epsilon and b_abs < epsilon and c_abs >= epsilon:
        ret = func_init_deriv_1x0x1(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs >= epsilon and b_abs >= epsilon and c_abs < epsilon:
        ret = func_init_deriv_1x1x0(t, a, b, c, c1, c2, sigma, epsilon)

    elif a_abs >= epsilon and b_abs >= epsilon and c_abs >= epsilon:
        ret = func_init_deriv_1x1x1(t, a, b, c, c1, c2, sigma, epsilon)

    else:
        ret = func_init_deriv_1x1x1(t, a, b, c, c1, c2, sigma, epsilon)

    if fprime:
        dt = 1e-3
        f_dt = deu_horse_init_deriv(t + dt, a, b, c, c1, c2)
        fp =  [torch.clamp((rrdt - rr) / dt, min = -max_val,max=max_val) for rrdt,rr in zip(f_dt, ret)]  
        return [torch.clamp(rr, min = -max_val,max=max_val) for rr in ret] + fp
    else:
        return [torch.clamp(rr, min = -max_val,max=max_val) for rr in ret]

def func_0x0x1(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t1 = 0.1e1 / c
    t3 = cap_exp(-sigma * t)
    t4 = 1 + t3
    t5 = 0.1e1 / t4
    if not derivative:
        f = t5 * t1
        return f
    t6 = t4 ** 2
    fp = t3 * sigma / t6 * t1
    fa = torch.zeros_like(t)
    fb = torch.zeros_like(t)
    t10 = c ** 2
    fc = -t5 / t10
    fc1 = torch.zeros_like(t)
    fc2 = torch.zeros_like(t)
    return fp, fa, fb, fc, fc1, fc2


def func_0x1x0(t, a, b, c, c1, c2, derivative, sigma, epsilon):
        #print ("TATA0", t) 
        #print ("TATA1", t.size(), a.size(), b.size(), c.size(), c1.size(), c2.size())
    t1 = 0.1e1 / b
    t2 = Heaviside(t)  
    t3 = t2 * t1
        #print ("TATA2", a, b, c, c1, c2, t3, t2, t * t3 + c1)
    if not derivative:
        f = t * t3 + c1
        return f
    t6 = cap_exp(-sigma * t)
    t8 = (1 + t6) ** 2
    fp = t * t6 * sigma / t8 * t1 + t3
    fa = torch.zeros_like(t)
    t14 = b ** 2
    fb = -t * t2 / t14
        #print ("TATA3", a, b, c, c1, c2, t3, t2, t6, t8, fp, fb)
    fc = torch.zeros_like(t)
    fc1 = torch.ones_like(t)
    fc2 = torch.zeros_like(t)
    return fp, fa, fb, fc, fc1, fc2


def func_0x1x1(t, a, b, c, c1, c2, derivative, sigma, epsilon): 
    t1 = 0.1e1 / b
    t2 = t1 * c
    t4 = cap_exp(-t * t2)
    t5 = c1 * t4
    t6 = Heaviside(t)
    t7 = t4 - 1
    t8 = t7 * t6
    t9 = 0.1e1 / c
    if not derivative:
        f = -t9 * t8 + t5
        return f
    t13 = cap_exp(-sigma * t)
    t15 = (1 + t13) ** 2
    t21 = t1 * t6
    fp = -t5 * t2 - t9 * t7 * t13 * sigma / t15 + t4 * t21
    fa = torch.zeros_like(t)
    t23 = b ** 2
    t24 = 0.1e1 / t23
    t26 = t4 * t
    fb = c1 * t26 * t24 * c - t26 * t24 * t6
    t35 = c ** 2
    fc = -t5 * t * t1 + t9 * t26 * t21 + 0.1e1 / t35 * t8
    fc1 = t4
    fc2 = torch.zeros_like(t)
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x0(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t1 = 0.1e1 / a
    t2 = Heaviside(t)
    t3 = t2 * t1
    t4 = t ** 2
    if not derivative:
        f = t4 * t3 / 2 + c1 * t + c2
        return f
    t9 = cap_exp(-sigma * t)
    t11 = (1 + t9) ** 2
    fp = t4 * t9 * sigma / t11 * t1 / 2 + t * t3 + c1
    t19 = a ** 2
    fa = -t4 * t2 / t19 / 2
    fb = torch.zeros_like(t)
    fc = torch.zeros_like(t)
    fc1 = t
    fc2 = torch.ones_like(t)
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x1_a_neg_c_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t2 = torch.sqrt(c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = torch.sin(t5)
    t8 = torch.cos(t5)
    t10 = Heaviside(t)
    t11 = t8 - 1
    t12 = t11 * t10
    t13 = 0.1e1 / c
    if not derivative:
        f = c1 * t8 + c2 * t6 - t13 * t12
        return f
    t15 = c2 * t8
    t17 = c1 * t6
    t20 = cap_exp(-sigma * t)
    t22 = (1 + t20) ** 2
    fp = t15 * t4 - t17 * t4 - t13 * t11 * t20 * sigma / t22 + t13 * t6 * t3 * t2 * t10
    t32 = 0.1e1 / t2
    t37 = a ** 2
    t41 = t * c * t3 * t32 / 2 - t / t37 * t2
    fa = t13 * t6 * t41 * t10 - c1 * t6 * t41 + c2 * t8 * t41
    fb = torch.zeros_like(t)
    t49 = t * t32
    t59 = c ** 2
    fc = t15 * t49 / 2 - t17 * t49 / 2 + t13 * t6 * t * t32 * t10 / 2 + 0.1e1 / t59 * t12
    fc1 = t8
    fc2 = t6
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x1_a_neg_c_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t2 = torch.sqrt(-c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = cap_exp(t5)
    t7 = c2 * t6
    t8 = cap_exp(-t5)
    t9 = c1 * t8
    t10 = Heaviside(t)
    t11 = torch.sqrt(c)
    t12 = torch.sqrt(-a)
    t13 = 0.1e1 / t12
    t14 = t13 * t11
    t15 = t * t14
    t16 = cap_exp(t15)
    t17 = cap_exp(-t15)
    t18 = t16 - 2 + t17
    t19 = t18 * t10
    t20 = 0.1e1 / c
    if not derivative:
        f = t7 + t9 - t20 * t19 / 2
        return f
    t26 = cap_exp(-sigma * t)
    t28 = (1 + t26) ** 2
    fp = t7 * t4 - t9 * t4 - t20 * t18 * t26 * sigma / t28 / 2 - t20 * (t16 * t14 - t17 * t14) * t10 / 2
    t41 = 0.1e1 / t2
    t46 = a ** 2
    t50 = -t * c * t3 * t41 / 2 - t / t46 * t2
    t58 = -0.1e1 / t12 / a * t11
    t59 = t16 * t
    t61 = t17 * t
    fa = c2 * t6 * t50 - c1 * t8 * t50 - t20 * (t59 * t58 / 2 - t61 * t58 / 2) * t10 / 2
    fb = torch.zeros_like(t)
    t68 = t * t41
    t72 = t13 / t11
    t79 = c ** 2
    fc = -t7 * t68 / 2 + t9 * t68 / 2 - t20 * (t59 * t72 / 2 - t61 * t72 / 2) * t10 / 2 + 0.1e1 / t79 * t19 / 2
    fc1 = t8
    fc2 = t6
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x1_a_pos_c_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t2 = torch.sqrt(-c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = cap_exp(t5)
    t7 = c2 * t6
    t8 = cap_exp(-t5)
    t9 = c1 * t8
    t10 = Heaviside(t)
    t11 = torch.sqrt(a)
    t12 = 0.1e1 / t11
    t13 = torch.sqrt(-c)
    t14 = t13 * t12
    t15 = t * t14
    t16 = cap_exp(t15)
    t17 = cap_exp(-t15)
    t18 = t16 - 2 + t17
    t19 = t18 * t10
    t20 = 0.1e1 / c
    if not derivative:
        f = t7 + t9 - t20 * t19 / 2
        return f
    t26 = cap_exp(-sigma * t)
    t28 = (1 + t26) ** 2
    fp = t7 * t4 - t9 * t4 - t20 * t18 * t26 * sigma / t28 / 2 - t20 * (t16 * t14 - t17 * t14) * t10 / 2
    t41 = 0.1e1 / t2
    t46 = a ** 2
    t50 = -t * c * t3 * t41 / 2 - t / t46 * t2
    t58 = t13 / t11 / a
    t59 = t16 * t
    t61 = t17 * t
    fa = c2 * t6 * t50 - c1 * t8 * t50 - t20 * (-t59 * t58 / 2 + t61 * t58 / 2) * t10 / 2
    fb = torch.zeros_like(t)
    t68 = t * t41
    t72 = 0.1e1 / t13 * t12
    t79 = c ** 2
    fc = -t7 * t68 / 2 + t9 * t68 / 2 - t20 * (-t59 * t72 / 2 + t61 * t72 / 2) * t10 / 2 + 0.1e1 / t79 * t19 / 2
    fc1 = t8
    fc2 = t6
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x1_a_pos_c_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t2 = torch.sqrt(c * a)
    t3 = 0.1e1 / a
    t4 = t3 * t2
    t5 = t * t4
    t6 = torch.sin(t5)
    t8 = torch.cos(t5)
    t10 = Heaviside(t)
    t11 = torch.sqrt(c)
    t12 = torch.sqrt(a)
    t13 = 0.1e1 / t12
    t15 = t * t13 * t11
    t16 = torch.cos(t15)
    t17 = t16 - 1
    t18 = t17 * t10
    t19 = 0.1e1 / c
    if not derivative:
        f = c1 * t8 + c2 * t6 - t19 * t18
        return f
    t21 = c2 * t8
    t23 = c1 * t6
    t26 = cap_exp(-sigma * t)
    t28 = (1 + t26) ** 2
    t35 = 0.1e1 / t11 * t10
    t36 = torch.sin(t15)
    fp = t21 * t4 - t23 * t4 - t19 * t17 * t26 * sigma / t28 + t36 * t13 * t35
    t39 = 0.1e1 / t2
    t44 = a ** 2
    t48 = t * c * t3 * t39 / 2 - t / t44 * t2
    fa = c2 * t8 * t48 - c1 * t6 * t48 - t36 * t / t12 / a * t35 / 2
    fb = torch.zeros_like(t)
    t59 = t * t39
    t71 = c ** 2
    fc = t21 * t59 / 2 - t23 * t59 / 2 + t36 * t * t13 / t11 / c * t10 / 2 + 0.1e1 / t71 * t18
    fc1 = t8
    fc2 = t6
    return fp, fa, fb, fc, fc1, fc2


def func_1x1x0(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t1 = b ** 2
    t2 = 1 / t1
    t3 = 0.1e1 / a
    t6 = cap_exp(-t * t3 * b)
    t7 = t6 * t2
    t8 = Heaviside(t)
    t9 = a * t8
    t11 = t8 * t2
    t13 = 1 / b
    t14 = t6 * t13
    t15 = c1 * a
    t17 = t8 * t13
    if not derivative:
        f = -a * t11 + t * t17 - t15 * t14 + t9 * t7 + c2
        return f
    t21 = cap_exp(-sigma * t)
    t23 = (1 + t21) ** 2
    t24 = 0.1e1 / t23
    t26 = t21 * sigma
    t27 = a * t26
    t31 = c1 * t6
    fp = t * t26 * t24 * t13 - t27 * t24 * t2 + t27 * t24 * t7 - t6 * t17 + t17 + t31
    fa = t8 * t6 * t * t3 * t13 - t31 * t * t3 - c1 * t14 + t8 * t7 - t11
    t44 = 1 / t1 / b
    fb = -t8 * t6 * t * t2 + 2 * a * t8 * t44 + t31 * t * t13 - 2 * t9 * t6 * t44 - t * t11 + t15 * t7
    fc = torch.zeros_like(t)
    fc1 = -a * t14
    fc2 = torch.ones_like(t)
    return fp, fa, fb, fc, fc1, fc2


def func_1x1x1_Delta_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t1 = 0.1e1 / a
    t2 = t1 * b
    t5 = cap_exp(-t * t2 / 2)
    t8 = b ** 2
    t9 = 4 * a * c - t8
    t10 = torch.sqrt(t9)
    t13 = t * t1 * t10 / 2
    t14 = torch.sin(t13)
    t15 = t14 * t5
    t16 = c2 * t15
    t17 = torch.cos(t13)
    t18 = t17 * t5
    t19 = c1 * t18
    t20 = Heaviside(t)
    t21 = t18 - 1
    t23 = t14 * b
    t25 = t10 * t21 + t5 * t23
    t26 = t25 * t20
    t27 = 0.1e1 / t10
    t28 = 1 / c
    t29 = t28 * t27
    if not derivative:
        f = -t29 * t26 + t16 + t19
        return f
    t33 = t10 * t5
    t34 = t17 * t1
    t40 = t14 * t1
    t45 = cap_exp(-sigma * t)
    t47 = (1 + t45) ** 2
    t63 = t14 * t8
    fp = -t16 * t2 / 2 + c2 * t34 * t33 / 2 - t19 * t2 / 2 - c1 * t40 * t33 / 2 - t28 * t27 * t25 * t45 * sigma / t47 - t29 * (t10 * (-t18 * t2 / 2 - t40 * t33 / 2) + t5 * t34 * t10 * b / 2 - t5 * t1 * t63 / 2) * t20
    t70 = a ** 2
    t71 = 1 / t70
    t72 = t71 * b
    t73 = t * t72
    t82 = t * c * t1 * t27 - t * t71 * t10 / 2
    t83 = t82 * t5
    t84 = c2 * t17
    t88 = c1 * t14
    t90 = t5 * t
    t91 = t17 * t90
    t97 = t27 * t21
    t110 = 0.1e1 / t10 / t9
    fa = t16 * t73 / 2 + t84 * t83 + t19 * t73 / 2 - t88 * t83 - t29 * (t10 * (t91 * t72 / 2 - t14 * t83) + 2 * c * t97 + t18 * t82 * b + t5 * t * t71 * t63 / 2) * t20 + 2 * t110 * t26
    t113 = t * t1
    t116 = t27 * t5
    t117 = t1 * t116
    t118 = b * t
    t144 = t28 * t110
    fb = -t16 * t113 / 2 - t84 * t118 * t117 / 2 - t19 * t113 / 2 + t88 * t118 * t117 / 2 - t29 * (t10 * (t14 * t118 * t117 / 2 - t18 * t113 / 2) - b * t97 + t15 - t91 * t1 * t27 * t8 / 2 - t5 * t113 * t23 / 2) * t20 - b * t144 * t26
    t164 = c ** 2
    fc = c2 * t17 * t * t116 - c1 * t14 * t * t116 - t29 * (t91 * t27 * b + 2 * a * t97 - t14 * t90) * t20 + 2 * a * t144 * t26 + 1 / t164 * t27 * t26
    fc1 = t18
    fc2 = t15
        #print ("F ret", fp.type(), fa.type())
    return fp, fa, fb, fc, fc1, fc2


def func_1x1x1_Delta_eq_0(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t1 = 1 / b
    t2 = c * t1
    t5 = cap_exp(-2 * t * t2)
    t6 = c2 * t5
    t7 = t * t5
    t8 = c1 * t7
    t9 = c * t
    t11 = 2 * t9 + b
    t13 = t5 * t11 - b
    t14 = Heaviside(t)
    t15 = t14 * t13
    t16 = 1 / c
    t17 = t16 * t1
    if not derivative:
        f = -t17 * t15 + t6 + t8
        return f
    t23 = t5 * c1
    t24 = t5 * c
    t25 = t1 * t11
    t32 = cap_exp(-sigma * t)
    t34 = (1 + t32) ** 2
    fp = -2 * t6 * t2 - 2 * t8 * t2 + t23 - t17 * t14 * (-2 * t24 * t25 + 2 * t24) - t16 * t1 * t32 * sigma / t34 * t13
    t41 = b ** 2
    t43 = 1 / t41 / b
    t44 = c ** 2
    t45 = t44 * t43
    t46 = c2 * t7
    t49 = t ** 2
    t51 = c1 * t5 * t49
    t66 = 1 / t41
    fa = 8 * t46 * t45 + 8 * t51 * t45 - 2 * t66 * (2 * t5 + 8 * t5 * t * t44 * t43 * (b * t + t16 * t41 / 2) - 2) * t14
    t69 = c * t66
    fb = 2 * t46 * t69 + 2 * t51 * t69 - t17 * t14 * (2 * t5 * t9 * t66 * t11 + t5 - 1) + t16 * t66 * t15
    fc = -2 * t6 * t * t1 - 2 * t23 * t49 * t1 - t17 * t14 * (-2 * t7 * t25 + 2 * t7) + 1 / t44 * t1 * t15
    fc1 = t7
    fc2 = t5
        #print ("F ret", fp.type(), fa.type())
    return fp, fa, fb, fc, fc1, fc2


def func_1x1x1_Delta_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    t3 = b ** 2
    t4 = -4 * a * c + t3
    t5 = torch.sqrt(t4)
    t6 = -b + t5
    t7 = 1 / a
    t8 = t7 * t6
    t11 = cap_exp(t * t8 / 2)
    t12 = c2 * t11
    t13 = b + t5
    t14 = t * t13
    t17 = cap_exp(-t7 * t14 / 2)
    t18 = c1 * t17
    t19 = Heaviside(t)
    t23 = t11 * t13 + t17 * t6 - 2 * t5
    t24 = t23 * t19
    t25 = 0.1e1 / t5
    t26 = 1 / c
    t27 = t26 * t25
    if not derivative:
        f = t12 + t18 - t27 * t24 / 2
        return f
    t34 = cap_exp(-sigma * t)
    t36 = (1 + t34) ** 2
    t43 = t6 * t13
    t46 = t17 * t7
    fp = t12 * t8 / 2 - t18 * t7 * t13 / 2 - t26 * t25 * t23 * t34 * sigma / t36 / 2 - t27 * (t11 * t7 * t43 / 2 - t46 * t43 / 2) * t19 / 2
    t53 = c * t25
    t54 = t * t7
    t55 = t54 * t53
    t56 = a ** 2
    t57 = 1 / t56
    t61 = -t55 - t * t57 * t6 / 2
    t66 = t55 + t57 * t14 / 2
    t83 = 0.1e1 / t5 / t4
    fa = c2 * t11 * t61 + c1 * t17 * t66 - t27 * (t11 * t61 * t13 + t17 * t66 * t6 - 2 * t11 * t53 - 2 * t17 * t53 + 4 * t53) * t19 / 2 - t83 * t24
    t85 = b * t25
    t86 = -1 + t85
    t88 = t11 * t
    t91 = 1 + t85
    t109 = t26 * t83
    fb = c2 * t88 * t7 * t86 / 2 - c1 * t46 * t * t91 / 2 - t27 * (t11 * t91 + t11 * t54 * t86 * t13 / 2 + t17 * t86 - t17 * t54 * t91 * t6 / 2 - 2 * t85) * t19 / 2 + b * t109 * t24 / 2
    t113 = t * t25
    t116 = a * t25
    t133 = c ** 2
    fc = -t12 * t113 + t18 * t113 - t27 * (t17 * t * t25 * t6 - t88 * t25 * t13 - 2 * t11 * t116 - 2 * t17 * t116 + 4 * t116) * t19 / 2 - a * t109 * t24 + 1 / t133 * t25 * t24 / 2
    fc1 = t17
    fc2 = t11
        #print ("F ret", fp.type(), fa.type())
    return fp, fa, fb, fc, fc1, fc2


def func_1x0x1(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    if a < 0 and c < 0:
        return func_1x0x1_a_neg_c_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    if a < 0 and c > 0:
        return func_1x0x1_a_neg_c_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    if a > 0 and c < 0:
        return func_1x0x1_a_pos_c_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    if a > 0 and c > 0:
        return func_1x0x1_a_pos_c_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    

def func_1x1x1(t, a, b, c, c1, c2, derivative, sigma, epsilon):
    Delta = b ** 2 - 4 * a * c
    
    if torch.abs(Delta) < epsilon:
        #b = torch.sign(b) * torch.sqrt(4 * a * c)
        a_sign = torch.sign(a)
        b_abs = torch.abs(b)
        c_sign = torch.sign(c)
        c = b_abs * c_sign / 2.
        a = b_abs * a_sign / 2.
        
        return func_1x1x1_Delta_eq_0(t, a, b, c, c1, c2, derivative, sigma, epsilon)
         
    if Delta < 0:
        return func_1x1x1_Delta_neg(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    if Delta > 0:
        return func_1x1x1_Delta_pos(t, a, b, c, c1, c2, derivative, sigma, epsilon)
    

def deu_horse(t, a, b, c, c1, c2, derivative=False, sigma=100, epsilon=.01, max_val=1000., gravitation_zone_percent=20):
    
    a_abs = torch.abs(a)
    b_abs = torch.abs(b)
    c_abs = torch.abs(c)
    
            
    if a_abs < epsilon and b_abs < epsilon and c_abs < epsilon:
        if derivative:
            ret = [torch.ones_like(t), torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t)]
        else:
            ret = t


    elif a_abs < epsilon and b_abs < epsilon and c_abs >= epsilon:
        ret = func_0x0x1(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs < epsilon and b_abs >= epsilon and c_abs < epsilon:
        ret = func_0x1x0(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs < epsilon and b_abs >= epsilon and c_abs >= epsilon:
        ret = func_0x1x1(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs >= epsilon and b_abs < epsilon and c_abs < epsilon:
        ret = func_1x0x0(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs >= epsilon and b_abs < epsilon and c_abs >= epsilon:
        ret = func_1x0x1(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs >= epsilon and b_abs >= epsilon and c_abs < epsilon:
        ret = func_1x1x0(t, a, b, c, c1, c2, derivative, sigma, epsilon)

    elif a_abs >= epsilon and b_abs >= epsilon and c_abs >= epsilon:
        ret = func_1x1x1(t, a, b, c, c1, c2, derivative, sigma, epsilon)
                #if derivative:
                #  print ("F ret",ret[0].type())

    else:
        ret = func_1x1x1(t, a, b, c, c1, c2, derivative, sigma, epsilon)


    if derivative:
        return [torch.clamp(rr, min = -max_val,max=max_val) for rr in ret]
    else:
        return torch.clamp(ret, min = -max_val,max=max_val)

def get_adjusted_derivative(t, a, b, c, c1, c2, epsilon, max_val):
   # a_new, b_new, c_new, c1_tilde, c2_tilde, a_old, b_old, c_old  = adjust_initial_cond(t, a, b, c, c1, c2, epsilon=epsilon, max_val=max_val)
    a_new, b_new, c_new, c1_tilde, c2_tilde, a_old, b_old, c_old  = adjust_initial_cond(t, a, b, c, c1, c2, epsilon=epsilon, max_val=max_val)

    fp, fa, fb, fc, fc1, fc2 = deu_horse(t, a, b, c, c1, c2, derivative=True, max_val=max_val)
    
    b_adjusted = 0 
    if c1 == c1_tilde and c2 == c2_tilde:
        #return fp, fa, fb, fc, fc1, fc2, b_adjusted, c1, c2
        return fp, fa, fb, fc, fc1, fc2
    #zz = torch.zeros_like(t)
    fp_upd, fa_upd, fb_upd, fc_upd, fc1_upd, fc2_upd = deu_horse(t, a_new, b_new, c_new, c1_tilde, c2_tilde, derivative=True, max_val=max_val)
    #pdb.set_trace()
    if torch.abs(a - a_new) > 1e-9:
        #if torch.abs(a_old - a_new) < .1 * epsilon:
        #    return fp, fa_upd, zz, zz, fc1, fc2, b_adjusted, c1, c2
        fa = a_old * fa_upd / a_new
        fa = fa_upd
        
    if torch.abs(b_old - b_new) > 1e-9:
        #if torch.abs(b - b_new) < .1 * epsilon:
        #    return fp, zz, fb_upd, zz, fc1, fc2, b_adjusted, c1, c2
        fb = b_old * fb_upd / b_new
        fb = fb_upd
        #b_adjusted |= 2
        #fc1, fc2 = fc1 * 0, fc2 * 0
    if torch.abs(c_old - c_new) > 1e-9:
        #if torch.abs(c - c_new) < .1 * epsilon:
        #    return fp, zz, zz, fc_upd, fc1, fc2, b_adjusted, c1, c2
        fc = c_old * fc_upd / c_new
        fc = fc_upd
        #b_adjusted |= 4
        #fc1, fc2 = fc1 * 0, fc2 * 0

    #return fp, fa, fb, fc, fc1, fc2, b_adjusted, c1_tilde, c2_tilde
    return fp, fa, fb, fc, fc1, fc2

def deu_activation(t, a, b, c, c1, c2, derivative=False, sigma=100, epsilon=.01, max_val=1000.):
    #pdb.set_trace()
    if not derivative:
        return deu_horse(t, a, b, c, c1, c2, derivative=False, sigma=sigma, epsilon=epsilon, max_val=max_val)
    else:
        return get_adjusted_derivative(t, a, b, c, c1, c2, epsilon, max_val)

def adjust_initial_cond(t, a, b, c, c1, c2, epsilon, max_val):
    if torch.abs(a) >= epsilon and torch.abs(b) >= epsilon and torch.abs(c) >= epsilon:
        return a, b, c, c1, c2,  a, b, c
    t = t.mean()
    f = deu_horse(t.mean(), a, b, c, c1, c2, max_val=max_val)
    fp, fa, fb, fc, fc1, fc2 = deu_horse(t.mean(), a, b, c, c1, c2, derivative=True)
    if torch.abs(a) < epsilon:
        if a == 0.:
            a = (torch.rand(1)) * epsilon * .07
        a_new = epsilon * torch.sign(a)
    else:
        a_new = a
    if torch.abs(b) < epsilon:
        if b == 0.:
            b = (torch.rand(1)) * epsilon * .07
        b_new = epsilon * torch.sign(b)
    else:
        b_new = b
    if torch.abs(c) < epsilon:
        if c == 0.:
            c = (torch.rand(1)) * epsilon * .07
        c_new = epsilon * torch.sign(c)
    else:
        c_new = c       
    f_homo = deu_horse(t.mean(), a_new, b_new, c_new, 0, 0, max_val=max_val)
    fp_homo, fa_homo, fb_homo, fc_homo, fc1_homo, fc2_homo = deu_horse(t.mean(), a_new, b_new, c_new, 0, 0, derivative=True, max_val=max_val)
    dc1, dc2, dc1_p, dc2_p = deu_horse_init_deriv(t.mean(), a_new, b_new, c_new, 0, 0, fprime=True, sigma=100, epsilon=epsilon, max_val=max_val)
    #dc1_h, dc2_h, dc1_p_h, dc2_p_h = deu_horse_init_deriv(t.mean(), a_new, b_new, c_new, c1, c2, fprime=True, sigma=100, epsilon=epsilon, max_val=max_val)

    ff = np.mat([[f - f_homo], [fp - fp_homo]])

    AA = np.mat([[dc1, dc2], [dc1_p, dc2_p]])
    
    
    #np.linalg.solve(AA, ff)
    
    CC = np.linalg.inv(AA.T*AA + np.eye(2)* 1e-9) * AA.T * ff
    
    c1_tilde = torch.FloatTensor([CC[0,0]])
    c2_tilde = torch.FloatTensor([CC[1,0]])
    
    return a_new, b_new, c_new, c1_tilde, c2_tilde, a, b, c

    """print(fp_homo + c1 * dc1_p + c2 * dc2_p - fp)
    print(f_homo + c1 * dc1 + c2 * dc2 - f)
    
    
    print(torch.abs(fp_homo + c1 * dc1_p + c2 * dc2_p - fp).sum())
    print(torch.abs(f_homo + c1 * dc1 + c2 * dc2 - f).sum())
    
    
    print("a:", a, "b:", b, "c:", c, "c1:", c1, "c2:", c2)
    display_save_fig(a, b, c, c1, c2, max_val=2000)
    display_save_fig(a, b, c, 0, 0, max_val=2000)
    
    pdb.set_trace()
    if torch.abs(f_homo + c1 * fc1 + c2 * fc2 - f).sum() > 1.:
        display_save_fig(a, b, c, c1, c2)
        display_save_fig(a, b, c, 0, 0)
        pdb.set_trace()"""

def display_save_fig(a, b, c, c1, c2, b_save=False, max_val = 1000):
    import matplotlib.pylab as plt
    import pdb
    import numpy as np
    t = torch.FloatTensor(np.linspace(-2,5, 1000))
    #x = torch.FloatTensor(np.linspace(-1,9, 20))
    
    #pdb.set_trace()
    f = deu_activation(t, a, b, c, c1, c2)
    #
    fp, fa, fb, fc, fc1, fc2, b_adjusted, c1_tilde, c2_tilde = deu_activation(t, a, b, c, c1, c2, derivative=True, max_val=max_val)
    
    #pdb.set_trace()
    print("a:", a, "b:", b, "c:", c, "c1:", c1, "c2:", c2)
    plt.plot(t.numpy(),(f).numpy(),linewidth=3)
    plt.plot(t.numpy(),(fp).numpy(),linewidth=2)
    plt.plot(t.numpy(),(fa).numpy(),linestyle='dashed')
    plt.plot(t.numpy(),(fb).numpy(),linestyle='dashed')
    plt.plot(t.numpy(),(fc).numpy(),linestyle='dashed')
    plt.legend(["y(t)  a=%.2f,b=%.2f,c=%.2f"%(a,b,c), "y'(t)  c1=%.2f.c2=%.2f"%(c1,c2), "y_a(t)", "y_b(t)", "y_c(t)"])
    if b_save:
        file_name = "a_%.2f_b_%.2f_c_%.2f_c1_%.2f_c2_%.2f"%(a,b,c,c1,c2)
        file_name = file_name.replace('.','_')
        plt.savefig("./charts/"+file_name+".pdf")
    plt.show()
    
def unit_test1():

    import numpy as np
    a = torch.FloatTensor([1])
    b = torch.FloatTensor([-1])
    c = torch.FloatTensor([1])
    c1, c2 = torch.zeros(2)
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])

    
    a = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    b = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c1 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c2 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])

    a = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
    b = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
    c = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
    c1 = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
    c2 = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])

    a = torch.FloatTensor([-3.3])
    b = torch.FloatTensor([1.6])
    c = torch.FloatTensor([-1.])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    
    a = torch.FloatTensor([1.])
    b = torch.FloatTensor([0.])
    c = torch.FloatTensor([0.])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    
    a = torch.FloatTensor([.8])
    b = torch.FloatTensor([0.5])
    c = torch.FloatTensor([0.4])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    
    a = torch.FloatTensor([3.])
    b = torch.FloatTensor([-1.])
    c = torch.FloatTensor([-.5])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)


    a = torch.FloatTensor([1])
    b = torch.FloatTensor([0.])
    c = torch.FloatTensor([5.])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([2.])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    
    a = torch.FloatTensor([.2])
    b = torch.FloatTensor([.2])
    c = torch.FloatTensor([.8])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    
    a = torch.FloatTensor([0])
    b = torch.FloatTensor([1])
    c = torch.FloatTensor([0])
    c1 = torch.FloatTensor([0])
    c2 = torch.FloatTensor([0])
    display_save_fig(a, b, c, c1, c2, b_save=True)
    

def unit_test2():

    import numpy as np

    
    a = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    b = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c1 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c2 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    
    for i in range(10):
        a = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/10.])
        b = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/10.])
        c = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/10.])*0
        c1 = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
        c2 = torch.FloatTensor([int(100 * (np.random.uniform() - .5))/100.])
        display_save_fig(a, b, c, c1, c2, b_save=False)


def unit_test3():

    import numpy as np
    import pdb 
    a = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    b = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c1 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    c2 = torch.FloatTensor([(np.random.uniform() > .5) * 2 * (np.random.uniform() - .5)])
    a = torch.FloatTensor([-3.])
    b = torch.FloatTensor([-2.])
    c = torch.FloatTensor([-1.])
    c1 = torch.FloatTensor([-0.5])
    c2 = torch.FloatTensor([-0.5])
    
    
    for i in range(100):
        eps = 1e-2
        ae = torch.FloatTensor([int(100 * (np.random.uniform()))/10. - 4.5]) * eps
        be = torch.FloatTensor([int(100 * (np.random.uniform()))/10. - 4.5]) * eps
        ce = torch.FloatTensor([int(100 * (np.random.uniform()))/10. - 4.5]) * eps
        c1e = torch.FloatTensor([int(100 * (np.random.uniform()))/10. - 4.5]) * eps
        c2e = torch.FloatTensor([int(100 * (np.random.uniform()))/10. - 4.5]) * eps
        a += ae 
        b += be 
        c += ce 
        c1 += c1e 
        c2 += c2e 
        display_save_fig(a, b, c, c1, c2, b_save=False) 
        #pdb.aset_trace()
import pdb
def unit_test_initial_cond_distribution():
    n_tests = 10000
    import numpy as np
    L = .4
    d = {}
    d["0x0x0"] = 0 
    d["0x0x0"] = 0
    d["0x0x1"] = 0
    d["0x1x0"] = 0
    d["0x1x1"] = 0
    d["1x0x0"] = 0
    d["1x0x1"] = 0
    d["1x1x0"] = 0
    d["1x1x1"] = 0
    epsilon = .01
    for i in range(n_tests):  
        a = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        b = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        c = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        c1 = torch.FloatTensor([(L * (np.random.uniform() - .5))/100.])
        c2 = torch.FloatTensor([(L * (np.random.uniform() - .5))/100.])
        a_abs = torch.abs(a)
        b_abs = torch.abs(b)
        c_abs = torch.abs(c)
        
        #print("a:", a, "b:", b, "c:", c, "c1:", c1, "c2:", c2)
        t = torch.FloatTensor(np.linspace(-2,5, 1000))
        f = deu_horse(t, a, b, c, c1, c2)
        fp, fa, fb, fc, fc1, fc2 = deu_horse(t, a, b, c, c1, c2, derivative=True)
        dc1, dc2 = deu_horse_init_deriv(t, a, b, c, c1, c2, sigma=100, epsilon=epsilon, max_val=1000.)

        if torch.abs(dc1-fc1).sum() + torch.abs(dc2-fc2).sum() > 0:
            pdb.set_trace()
        if a_abs < epsilon and b_abs < epsilon and c_abs < epsilon:
            d["0x0x0"] += 1
        elif a_abs < epsilon and b_abs < epsilon and c_abs >= epsilon:
            d["0x0x1"] += 1

        elif a_abs < epsilon and b_abs >= epsilon and c_abs < epsilon:
            d["0x1x0"] += 1

        elif a_abs < epsilon and b_abs >= epsilon and c_abs >= epsilon:
            d["0x1x1"] += 1

        elif a_abs >= epsilon and b_abs < epsilon and c_abs < epsilon:
            d["1x0x0"] += 1

        elif a_abs >= epsilon and b_abs < epsilon and c_abs >= epsilon:
            d["1x0x1"] += 1

        elif a_abs >= epsilon and b_abs >= epsilon and c_abs < epsilon:
            d["1x1x0"] += 1

        elif a_abs >= epsilon and b_abs >= epsilon and c_abs >= epsilon:
            d["1x1x1"] += 1
        else:
            d["1x1x1"] += 1
    print(d)
    #pdb.set_trace()
    print(n_tests)
    print(np.sum(list(d.values())))



import pdb
def unit_test_initial_cond():
    n_tests = 1000
    import numpy as np
    L = 3
    epsilon = .01
    for i in range(n_tests):  
        a = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        b = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        c = torch.FloatTensor([(L * (np.random.uniform() - .5))/10.])
        c1 = torch.FloatTensor([(L * (np.random.uniform() - .5))/1.])
        c2 = torch.FloatTensor([(L * (np.random.uniform() - .5))/1])

        #a, b, c, c1, c2 = [torch.FloatTensor([0.0217]), torch.FloatTensor([0.0136]), torch.FloatTensor([-0.0421]), torch.FloatTensor([-0.7048]), torch.FloatTensor([-0.3025])]
       # display_save_fig(a, b, c, c1, c2, max_val=2000)
        
        #pdb.set_trace()
        a_abs = torch.abs(a)
        b_abs = torch.abs(b)
        c_abs = torch.abs(c)


        t = torch.FloatTensor(np.linspace(-1,2, 10))
        a_new, b_new, c_new, c1_tilde, c2_tilde, a_old, b_old, c_old = adjust_initial_cond(t, a, b, c, c1, c2, epsilon=epsilon, max_val=2000)

        if c1 != c1_tilde or c2 != c2_tilde:
            def nnn(a, b, c, epsilon):
                if torch.abs(a) < epsilon:
                    a_new = epsilon * torch.sign(a)
                else:
                    a_new = a
                if torch.abs(b) < epsilon:
                    b_new = epsilon * torch.sign(b)
                else:
                    b_new = b
                if torch.abs(c) < epsilon:
                    c_new = epsilon * torch.sign(c)
                else:
                    c_new = c
                return a_new, b_new, c_new

            a_new, b_new, c_new = nnn(a, b, c, epsilon)            
            print(deu_horse(t.mean(), a, b, c, c1, c2))
            print(deu_horse(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde))
            print(deu_horse(t.mean(), a, b, c, c1, c2, derivative=True))
            print(deu_horse(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde, derivative=True))
            A = [(i.data.numpy(), j.data.numpy())  for i,j in zip(deu_horse(t.mean(), a, b, c, c1, c2, derivative=True), deu_horse(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde, derivative=True))]
            #B = [(i[0][0], i[1][0]) for i in A]
            print("a:", a, "b:", b, "c:", c, "c1:", c1, "c2:", c2)
            print("c1~:", c1_tilde, "c2~:", c2_tilde)
            print(get_adjusted_derivative(t.mean(), a, b, c, c1, c2, epsilon, max_val=1000))
            print(adjust_initial_cond(t, a, b, c, c1, c2, epsilon, max_val=1000))
            pdb.set_trace()        
            print(deu_activation(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde))
            print(deu_activation(t.mean(), a, b, c, c1, c2))
            print(deu_activation(t.mean(), a, b, c, c1, c2, derivative=True))
            print(deu_activation(t.mean(), a, b, c, c1, c2))
            print(deu_activation(t.mean(), a, b, c, c1, c2, derivative=True))
            print(deu_horse(t.mean(), a, b, c, c1, c2))
            print(deu_horse(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde))
            #print(deu_horse(t.mean(), a_new, b_new, c_new, c1_tilde, c2_tilde, derivative=True))
        
        #pdb.set_trace()


    
def main():
    #unit_test1()
    unit_test2()
    #unit_test3()
    #unit_test_initial_cond_distribution()
    #unit_test_initial_cond()
    
if __name__ == "__main__":
    main()
