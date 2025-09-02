TITLE RS membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Cortical regular spiking neuron
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-05
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX RSauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
   RANGE A_t, y
   RANGE a1, b1
   POINTER V_table, alpham_table, betam_table, alphah_table, betah_table, alphan_table, betan_table, pinf_table, taup_table, A_1_table, B_1_table
   RANGE V_val, alpham_val, betam_val, alphah_val, betah_val, alphan_val, betan_val, pinf_val, taup_val, A_1_val, B_1_val
   POINTER A_arr, Q_arr, amp1_arr, phi1_arr :, A1_arr, B1_arr
   RANGE A_s, Q_s, amp1_s, phi1_s :, A1_s, B1_s   
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.056 (S/cm2)
   ENa = 50.0 (mV)
   gKdbar = 0.006 (S/cm2)
   EK = -90.0 (mV)
   gMbar = 7.500000000000001e-05 (S/cm2)
   gLeak = 2.05e-05 (S/cm2)
   ELeak = -70.3 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   p : iM gate
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iM (mA/cm2)
   iLeak (mA/cm2)
   A_t (kPa)
   y
   a1 (nC/cm2)
   b1 (rad)
   q1 (nC/cm2)
   f1 (rad)
   V_table alpham_table betam_table alphah_table betah_table alphan_table betan_table pinf_table taup_table A_1_table B_1_table
   V_val (mV) alpham_val (/ms) betam_val (/ms) alphah_val (/ms) betah_val (/ms) alphan_val (/ms) betan_val (/ms) pinf_val (/ms) taup_val (/ms) A_1_val (nC/cm2) B_1_val (nC/cm2)
   A_arr Q_arr amp1_arr phi1_arr :A1_arr B1_arr
   A_s Q_s amp1_s phi1_s :A1_s B1_s   

}


INCLUDE "update2.inc"
INCLUDE "interp.inc"

FUNCTION fV() { 
VERBATIM
	double V_value;
	V_value = interp4D(_p_V_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1); //interp4D(_p_V_table, _p_A_arr, _p_Q_arr, _p_A1_arr, _p_B1_arr, A_s, Q_s, A1_s, B1_s, A_t, v, a1, b1);
	V_val = V_value;
	return(V_value);
ENDVERBATIM
	fV = V_value
}

FUNCTION falpham() { 
VERBATIM
	double alpham_value;
	alpham_value = interp4D(_p_alpham_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	alpham_val = alpham_value;
	return(alpham_value);
ENDVERBATIM
	falpham = alpham_value
}

FUNCTION fbetam() { 
VERBATIM
	double betam_value;
	betam_value = interp4D(_p_betam_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	betam_val = betam_value;
	return(betam_value);
ENDVERBATIM
	fbetam = betam_value
}

FUNCTION falphah() { 
VERBATIM
	double alphah_value;
	alphah_value = interp4D(_p_alphah_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	alphah_val = alphah_value;
	return(alphah_value);
ENDVERBATIM
	falphah = alphah_value
}

FUNCTION fbetah() { 
VERBATIM
	double betah_value;
	betah_value = interp4D(_p_betah_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	betah_val = betah_value;
	return(betah_value);
ENDVERBATIM
	fbetah = betah_value
}

FUNCTION falphan() { 
VERBATIM
	double alphan_value;
	alphan_value = interp4D(_p_alphan_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	alphan_val = alphan_value;
	return(alphan_value);
ENDVERBATIM
	falphan = alphan_value
}

FUNCTION fbetan() { 
VERBATIM
	double betan_value;
	betan_value = interp4D(_p_betan_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	betan_val = betan_value;
	return(betan_value);
ENDVERBATIM
	fbetan = betan_value
}

FUNCTION fpinf() { 
VERBATIM
	double pinf_value;
	pinf_value = interp4D(_p_pinf_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	pinf_val = pinf_value;
	return(pinf_value);
ENDVERBATIM
	fpinf = pinf_value
}

FUNCTION ftaup() { 
VERBATIM
	double taup_value;
	taup_value = interp4D(_p_taup_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	taup_val = taup_value;
	return(taup_value);
ENDVERBATIM
	ftaup = taup_value
}

FUNCTION fA_1() { 
VERBATIM
	double A_1_value;
	A_1_value = interp4D(_p_A_1_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	A_1_val = A_1_value;
	return(A_1_value);
ENDVERBATIM
	fA_1 = A_1_value
}

FUNCTION fB_1() { 
VERBATIM
	double B_1_value;
	B_1_value = interp4D(_p_B_1_table, _p_A_arr, _p_Q_arr, _p_amp1_arr, _p_phi1_arr, A_s, Q_s, amp1_s, phi1_s, A_t, v, q1, f1);
	B_1_val = B_1_value;
	return(B_1_value);
ENDVERBATIM
	fB_1 = B_1_value
}

INITIAL {
   update()
   m = falpham() / (falpham() + fbetam())
   h = falphah() / (falphah() + fbetah())
   n = falphan() / (falphan() + fbetan())
   p = fpinf()
}

BREAKPOINT {
   update()
   SOLVE states METHOD cnexp
   Vm = fV()
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iM = gMbar * p * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
   m' = falpham() * (1 - m) - fbetam() * m
   h' = falphah() * (1 - h) - fbetah() * h
   n' = falphan() * (1 - n) - fbetan() * n
   p' = (fpinf() - p) / ftaup()
}