"""Small independent checks of shkur_wid.pro; not an IDL execution or full port."""
import numpy as np

def internal_coated_reflection(g, m1, m2):
    # Original lines 467-483, zero coating thickness => phi=1.
    c = np.cos(g)
    c2 = np.sqrt(1 - np.sin(g)**2 / complex(m2)**2)
    c3 = np.sqrt(1 - np.sin(g)**2 / complex(m1)**2)
    a = (m2*c2-c)/(m2*c2+c)
    b = (c2-m2*c)/(c2+m2*c)
    d = (m1*c3-m2*c2)/(m1*c3+m2*c2)
    e = (m2*c3-m1*c2)/(m2*c3+m1*c2)
    return (abs((d+a)/(1+d*a))**2 + abs((e+b)/(1+e*b))**2)/2

g = np.pi/4
r = internal_coated_reflection(g, 1.5, 1.5)
print('zero-thickness internal R at 45 degrees:', r, '; expected: 1 (total internal reflection)')
assert abs(r - 0.05023991101223592) < 1e-12
assert g > np.arcsin(1/1.5)
original = complex(1.5**2+0.1**2, 1.5*0.1)
expected = complex(1.5, 0.1)**2
print('dielectric constant:', original, '; expected:', expected)
assert original != expected
ratio = 2*np.sqrt(np.log(2))/2.35
print('actual Gaussian FWHM / dlam:', ratio)
a = np.float32(1e4)
direct = a-np.sqrt(a*a-np.float32(1))
stable = 1/(float(a)+np.sqrt(float(a)**2-1))
print('albedo cancellation example, float32:', direct, '; stable:', stable)
assert direct == 0 and stable > 0
q = 1 - 1.0
rb, rf = q*0.1, q*0.5 + 1-q
print('porosity=1: numerator=', 1+rb**2-rf**2, '; denominator=', 2*rb)
assert rb == 0 and 1+rb**2-rf**2 == 0
print('All diagnostic checks reproduced the reported issues.')
