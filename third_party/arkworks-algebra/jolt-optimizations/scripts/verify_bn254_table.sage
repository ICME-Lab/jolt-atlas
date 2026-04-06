#!/usr/bin/sage
# vim: syntax=python
# vim: set ts=2 sw=2 et:

# Standalone script to generate BN254 G2 4D decomposition lookup table
# AND verify point equation for 100 random scalars
# This script contains all necessary functions inlined for standalone operation

import os

# Working directory
os.chdir(os.path.dirname(__file__))

from sage.structure.proof.all import arithmetic
arithmetic(False)

# Inlined curve constants for BN254_Snarks
def get_bn254_constants():
    """Get BN254 curve constants - inlined for standalone operation"""
    # BN254_Snarks parameters
    u = 0x44e992b44a6909f1
    p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
    t = 0x6f4d8248eeb859fbf83e9682e87cfd47
    
    lambda_psi = (t - 1) % r
    lattice = derive_lattice(r, lambda_psi, 4)
    return r, lambda_psi, lattice

def derive_lattice(r, lambdaR, m):
    """Derive lattice matrix for scalar decomposition"""
    lat = Matrix(matrix.identity(m))
    lat[0, 0] = r
    for i in range(1, m):
        lat[i, 0] = -lambdaR**i
    return lat.LLL()

def decompose_scalar_g2_lattice(k, lattice):
    """Lattice-based scalar decomposition"""
    k_vec = vector([k, 0, 0, 0])
    alpha = k_vec * lattice.inverse()
    alpha_rounded = vector([round(a) for a in alpha])
    short_vec = k_vec - alpha_rounded * lattice
    return tuple(short_vec)

def precompute_power_of_2_decompositions():
    """Precompute 4D decompositions for all powers of 2"""
    r, lambda_psi, lattice = get_bn254_constants()
    
    # Determine how many bits we need (up to log2(r))
    max_bits = int(r).bit_length()
    
    precomputed_table = {}
    
    for i in range(max_bits):
        power_of_2 = 1 << i  # 2^i
        k0, k1, k2, k3 = decompose_scalar_g2_lattice(power_of_2, lattice)
        
        # Handle negative values by storing absolute value + sign flag
        decomp = []
        neg_flags = []
        
        for val in [k0, k1, k2, k3]:
            if val < 0:
                decomp.append(-val)
                neg_flags.append(True)
            else:
                decomp.append(val)
                neg_flags.append(False)
        
        precomputed_table[i] = {
            'power': power_of_2,
            'decomp': decomp,
            'neg_flags': neg_flags,
            'original': [k0, k1, k2, k3]
        }
    
    return precomputed_table, max_bits

def decompose_by_precomputation(k, precomputed_table):
    """Decompose scalar k using precomputed powers of 2"""
    
    # Get binary representation of k
    k_bits = []
    temp_k = k
    bit_pos = 0
    
    while temp_k > 0:
        if temp_k & 1:
            k_bits.append(bit_pos)
        temp_k >>= 1
        bit_pos += 1
    
    # Sum decompositions for all set bits
    total_k0, total_k1, total_k2, total_k3 = 0, 0, 0, 0
    
    for bit_pos in k_bits:
        if bit_pos in precomputed_table:
            entry = precomputed_table[bit_pos]
            k0, k1, k2, k3 = entry['original']
            
            total_k0 += k0
            total_k1 += k1
            total_k2 += k2
            total_k3 += k3
    
    return total_k0, total_k1, total_k2, total_k3

def setup_bn254_g2():
    """Setup BN254 G2 curve and points for testing"""
    
    # BN254_Snarks parameters  
    p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
    
    # Setup Fp2 field 
    Fp = GF(p)
    R.<x> = PolynomialRing(Fp)
    Fp2.<i> = Fp.extension(x^2 + 1)  # BN254 uses i^2 = -1
    
    # G2 curve coefficient b' = 3/(9+i) on the twist
    # For BN254 D-twist: E': y^2 = x^3 + b/xi where xi = 9+i
    b_twist = 3 / (Integer(9) + i)
    
    # Create G2 curve
    G2 = EllipticCurve(Fp2, [0, b_twist])
    
    # Find a random point of order r on G2
    cofactor = G2.order() // r
    
    # Get a random point and multiply by cofactor to get order r
    attempts = 0
    while attempts < Integer(100):
        try:
            P_random = G2.random_point()
            P = cofactor * P_random
            if P != G2([0, 1, 0]) and r * P == G2([0, 1, 0]):
                return G2, P, Fp2
        except:
            pass
        attempts += 1
    
    raise Exception("Could not find valid G2 point")

def frobenius_psi_endomorphism_g2(P, G2, Fp2, k=1):
    """Apply Frobenius ψ (psi) endomorphism: untwist-Frobenius-twist"""
    if P.is_zero():
        return P
    
    x, y, z = P
    
    # Apply Frobenius map to coordinates (conjugation in Fp2)
    if (k & 1) == 1:
        # Odd power - apply conjugation
        x_frob = x.conjugate()
        y_frob = y.conjugate()
        z_frob = z.conjugate()
    else:
        # Even power - identity
        x_frob = x
        y_frob = y
        z_frob = z
    
    # Apply the Frobenius psi coefficients from bn254_snarks_frobenius.nim
    
    if k == 1:
        # For psi^1, we need coef=2 for x and coef=3 for y
        psi_coef2 = Fp2([Integer(0x2fb347984f7911f74c0bec3cf559b143b78cc310c2c3330c99e39557176f553d),
                         Integer(0x16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2)])
        psi_coef3 = Fp2([Integer(0x63cf305489af5dcdc5ec698b6e2f9b9dbaae0eda9c95998dc54014671a0135a),
                         Integer(0x7c03cbcac41049a0704b5a7ec796f2b21807dc98fa25bd282d37f632623b0e3)])
        
        x_result = x_frob * psi_coef2
        y_result = y_frob * psi_coef3
        z_result = z_frob
        
    elif k == 2:
        # For psi^2
        psi_coef2 = Fp2([Integer(0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48), 0])
        psi_coef3 = Fp2([Integer(0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46), 0])
        
        x_result = x_frob * psi_coef2
        y_result = y_frob * psi_coef3
        z_result = z_frob
        
    elif k == 3:
        # For psi^3
        psi_coef2 = Fp2([Integer(0x856e078b755ef0abaff1c77959f25ac805ffd3d5d6942d37b746ee87bdcfb6d),
                         Integer(0x4f1de41b3d1766fa9f30e6dec26094f0fdf31bf98ff2631380cab2baaa586de)])
        psi_coef3 = Fp2([Integer(0x2a275b6d9896aa4cdbf17f1dca9e5ea3bbd689a3bea870f45fcc8ad066dce9ed),
                         Integer(0x28a411b634f09b8fb14b900e9507e9327600ecc7d8cf6ebab94d0cb3b2594c64)])
        
        x_result = x_frob * psi_coef2
        y_result = y_frob * psi_coef3
        z_result = z_frob
        
    else:
        # For other powers, use identity
        x_result = x_frob
        y_result = y_frob  
        z_result = z_frob
        
    return G2([x_result, y_result, z_result])

def verify_point_equation_100_tests():
    """Verify s*P = k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P) for 100 random scalars"""
    
    # Get precomputed table and constants
    precomputed_table, _ = precompute_power_of_2_decompositions()
    r, lambda_psi, _ = get_bn254_constants()
    
    try:
        G2, P, Fp2 = setup_bn254_g2()
    except Exception as e:
        return False
    
    # Test with 100 random scalars
    for test_num in range(100):
        s = randint(1, r-1)
        
        try:
            # Compute s*P directly
            sP_direct = s * P
            
            # Get precomputed decomposition
            k0, k1, k2, k3 = decompose_by_precomputation(s, precomputed_table)
            
            # Verify algebraic reconstruction first
            reconstructed = (k0 + k1*lambda_psi + k2*lambda_psi^2 + k3*lambda_psi^3) % r
            if (s % r) != reconstructed:
                return False
            
            # Compute the 4D decomposition using Frobenius basis
            P0 = P                                            # P
            P1 = frobenius_psi_endomorphism_g2(P, G2, Fp2, k=1)  # φ(P)
            P2 = frobenius_psi_endomorphism_g2(P, G2, Fp2, k=2)  # φ²(P)
            P3 = frobenius_psi_endomorphism_g2(P, G2, Fp2, k=3)  # φ³(P)
            
            # Convert to signed representation for point arithmetic
            def make_signed(val, modulus):
                if val > modulus // 2:
                    return val - modulus
                return val
            
            k0_signed = make_signed(k0 % r, r)
            k1_signed = make_signed(k1 % r, r)
            k2_signed = make_signed(k2 % r, r)
            k3_signed = make_signed(k3 % r, r)
            
            # Compute k0*P0 + k1*P1 + k2*P2 + k3*P3
            sP_decomposed = k0_signed*P0 + k1_signed*P1 + k2_signed*P2 + k3_signed*P3
            
            # Check if the points are equal
            if sP_direct != sP_decomposed:
                return False
                
        except Exception as e:
            return False
    
    return True

# Main execution
if __name__ == "__main__":
    # Verify point equation for 100 random tests
    verification_passed = verify_point_equation_100_tests()
    
    # Exit with appropriate code based on verification result
    if verification_passed:
        exit(0)  # Success
    else:
        exit(1)  # Failure