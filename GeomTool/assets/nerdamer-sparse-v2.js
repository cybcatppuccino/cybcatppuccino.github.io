/* Nerdamer 2.0 sparse integer-polynomial factorization/GCD source subset; Apache-2.0. */
(function(root){'use strict';
const modules=Object.create(null),cache=Object.create(null);
let CAS_DEADLINE=Infinity;
function casBudgetCheck() {
    if (CAS_DEADLINE !== Infinity && Date.now() >= CAS_DEADLINE) throw Error('CAS factorization time budget exhausted');
}

function load(name){if(cache[name])return cache[name].exports;let make=modules[name];if(!make)throw Error('missing Nerdamer module '+name);let m={exports:{}};cache[name]=m;make(function(id){if(!id.startsWith('.'))throw Error('external module '+id);let x=name.split('/');x.pop();for(const seg of id.split('/')){if(seg==='..')x.pop();else if(seg!=='.')x.push(seg)}return load(x.join('/'));},m,m.exports);return m.exports;}
modules["core/classes/polynomial/SparsePolynomial"]=function(require,module,exports){
"use strict";
/**
 * @module sparsePolynomial
 *
 * Sparse polynomial storage with exact bigint coefficients and exponents.
 *
 * External references consulted for monomial representation and ordering:
 * - SymPy monomial operations:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/monomials.py
 * - SymPy lexicographic, graded lexicographic, and graded reverse lexicographic orders:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/orderings.py
 * - SymPy coefficient content and primitive-part operations:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/densetools.py
 * - SymPy sparse polynomial division:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/sparsetools.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SparsePolynomial = void 0;
const errors_1 = require("../../errors");
const bigint_1 = require("../../functions/bigint/bigint");
// CAS-only packed-monomial fast path. No rounding is ever permitted.
// The external sparse-polynomial API remains BigInt-based and unchanged.
const FAST_FALLBACK = Symbol('unsafe numeric intermediate');
const SAFE = BigInt(Number.MAX_SAFE_INTEGER);
function packTerms(poly, active) {
    if (active.length > 24) return null;
    const terms = [];
    let absoluteSum = 0n;
    const maximum = new Array(active.length).fill(0);
    for (const t of poly.termsByKey.values()) {
        const ac = t.coefficient < 0n ? -t.coefficient : t.coefficient;
        absoluteSum += ac;
        if (absoluteSum > SAFE) return null;
        const exponents = [];
        for (let j = 0; j < active.length; ++j) {
            const exp = t.exponents[active[j]];
            if (exp >= 65535n) return null;
            const value = Number(exp);
            exponents.push(value);
            if (value > maximum[j]) maximum[j] = value;
        }
        terms.push({ c: Number(t.coefficient), e: exponents });
    }
    return { terms, maximum, absoluteSum };
}
// Adapted from Accelerated's compact base-64 small-ring representation.
// At most eight *active* coordinates fit into 48 exact Number bits; large
// exponents and coefficients immediately use Optimized's general code below.
function geomBase64Multiply(a,b,active,x,y) {
    const powers=new Array(active.length);
    for(let i=0,v=1;i<active.length;i++,v*=64)powers[i]=v;
    function packed(terms) {
        return terms.map(t=>{
            let key=0;
            for(let i=0;i<active.length;i++)key+=t.e[i]*powers[i];
            return {key,c:t.c};
        });
    }
    const left=packed(x.terms),right=packed(y.terms),sums=new Map();
    for(const l of left)for(const r of right){
        const key=l.key+r.key;
        const c=(sums.get(key)||0)+l.c*r.c;
        if(c===0)sums.delete(key);else sums.set(key,c);
    }
    const result=SparsePolynomial.zero(a.variableCount);
    for(const [key,c] of sums){
        const e=Array(a.variableCount).fill(0n);
        let v=key;
        for(let i=0;i<active.length;i++){
            const digit=v%64;
            v=Math.floor(v/64);
            e[active[i]]=BigInt(digit);
        }
        Object.freeze(e);
        result.termsByKey.set(SparsePolynomial.keyOf(e),Object.freeze({coefficient:BigInt(c),exponents:e}));
    }
    return result;
}
function fastExactMultiply(a, b) {
    const active = [...new Set([...a.variables(), ...b.variables()])].sort((x,y)=>x-y);
    const x = packTerms(a,active), y = packTerms(b,active);
    if (x === null || y === null || x.absoluteSum * y.absoluteSum > SAFE) return FAST_FALLBACK;
    const n = active.length;
    for (let j = 0; j < n; ++j) if (x.maximum[j] + y.maximum[j] >= 65535) return FAST_FALLBACK;
    // Adaptive choice: numeric base-64 keys for small active rings; 16-bit
    // character keys for larger or higher-degree rings. No dimension limit on
    // the polynomial itself (the generic BigInt path remains available).
    if(n>0 && n<=8 && x.terms.length*y.terms.length>=128 &&
       x.maximum.every((e,i)=>e+y.maximum[i]<64))
       return geomBase64Multiply(a,b,active,x,y);
    const sums = new Map();
    for (const l of x.terms) for (const r of y.terms) {
        const ex = new Array(n);
        for (let j = 0; j < n; ++j) ex[j] = l.e[j] + r.e[j];
        const key = String.fromCharCode(...ex);
        const next = (sums.get(key) || 0) + l.c * r.c;
        if (next === 0) sums.delete(key);
        else sums.set(key, next);
    }
    const result = SparsePolynomial.zero(a.variableCount);
    for (const [key, c] of sums) {
        const ex = Array(a.variableCount).fill(0n);
        for(let i=0;i<key.length;++i)ex[active[i]]=BigInt(key.charCodeAt(i));
        Object.freeze(ex);
        result.termsByKey.set(SparsePolynomial.keyOf(ex), Object.freeze({coefficient: BigInt(c), exponents: ex}));
    }
    return result;
}
function comparePacked(a, b, order) {
    if (order !== 'lex') {
        let da = 0, db = 0;
        for (let i = 0; i < a.length; ++i) { da += a[i]; db += b[i]; }
        if (da !== db) return da - db;
    }
    if (order === 'grevlex') {
        for (let i = a.length - 1; i >= 0; --i) if (a[i] !== b[i]) return b[i] - a[i];
    } else {
        for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
}
function fastExactDivide(dividend, divisor, order) {
    const active = [...new Set([...dividend.variables(),...divisor.variables()])].sort((a,b)=>a-b);
    const x = packTerms(dividend,active), y = packTerms(divisor,active);
    if (x === null || y === null) return FAST_FALLBACK;
    const n = active.length;
    const dterms = y.terms;
    const dle = dterms.reduce((best, t) => !best || comparePacked(t.e, best.e, order) > 0 ? t : best, null);
    const terms = new Map();
    for (const t of x.terms) terms.set(String.fromCharCode(...t.e), t);
    const quotient = new Map();
    while (terms.size) {
        let lt = null;
        for (const term of terms.values()) {
            if (lt === null || comparePacked(term.e, lt.e, order) > 0) lt = term;
        }
        if (lt.c % dle.c !== 0) return null;
        const e = new Array(n);
        for (let i = 0; i < n; ++i) {
            const exp = lt.e[i] - dle.e[i];
            if (exp < 0) return null;
            e[i] = exp;
        }
        const c = lt.c / dle.c;
        if (!Number.isSafeInteger(c)) return FAST_FALLBACK;
        const qkey = String.fromCharCode(...e);
        const old = quotient.get(qkey);
        const nextq = (old?.c || 0) + c;
        if (!Number.isSafeInteger(nextq)) return FAST_FALLBACK;
        quotient.set(qkey, {c: nextq, e});
        for (const term of dterms) {
            const product = c * term.c;
            if (!Number.isSafeInteger(product)) return FAST_FALLBACK;
            const exp = new Array(n);
            for (let i = 0; i < n; ++i) {
                exp[i] = term.e[i] + e[i];
                if (exp[i] >= 65535) return FAST_FALLBACK;
            }
            const key = String.fromCharCode(...exp);
            const old = terms.get(key);
            const next = (old?.c || 0) - product;
            if (!Number.isSafeInteger(next)) return FAST_FALLBACK;
            if (next === 0) terms.delete(key);
            else terms.set(key, {c: next, e: old?.e || exp});
        }
    }
    const result = SparsePolynomial.zero(dividend.variableCount);
    for (const {c,e} of quotient.values()) {
        const ex = Array(dividend.variableCount).fill(0n);
        for(let i=0;i<e.length;++i)ex[active[i]]=BigInt(e[i]);
        Object.freeze(ex);
        result.termsByKey.set(SparsePolynomial.keyOf(ex), Object.freeze({coefficient: BigInt(c), exponents: ex}));
    }
    return result;
}

/**
 * Sparse polynomial with bigint coefficients and bigint exponents.
 *
 * The variable count is part of the polynomial rather than inferred from its support.
 * Exponent vectors therefore have one exact entry for every variable, including zero
 * polynomials and constants. The internal map key is an encoding detail and is never
 * exposed or parsed by polynomial algorithms.
 */
class SparsePolynomial {
    /**
     * Creates a sparse polynomial in a ring with a fixed number of variables.
     *
     * Duplicate monomials are combined and zero coefficients are removed.
     *
     * @param variableCount - Number of variables in the polynomial ring.
     * @param terms - Terms with exponent vectors of exactly variableCount entries.
     */
    constructor(variableCount, terms = []) {
        if (!Number.isSafeInteger(variableCount) || variableCount < 0) {
            throw new RangeError((0, errors_1.message)('sparseVariableCountInvalid'));
        }
        this.variableCount = variableCount;
        this.termsByKey = new Map();
        for (const term of terms) {
            SparsePolynomial.assertExponents(variableCount, term.exponents);
            const exponents = term.exponents.map(exponent => exponent);
            const frozenExponents = Object.freeze(exponents);
            const key = SparsePolynomial.keyOf(frozenExponents);
            const coefficient = (this.termsByKey.get(key)?.coefficient ?? 0n) + term.coefficient;
            if (coefficient === 0n) {
                this.termsByKey.delete(key);
            }
            else {
                this.termsByKey.set(key, Object.freeze({ coefficient, exponents: frozenExponents }));
            }
        }
    }
    /**
     * Verifies that an exponent vector belongs to a ring and contains only
     * non-negative exact integer exponents.
     */
    static assertExponents(variableCount, exponents) {
        if (exponents.length !== variableCount) {
            throw new RangeError((0, errors_1.message)('sparseExponentVectorLengthMismatch'));
        }
        for (const exponent of exponents) {
            if (typeof exponent !== 'bigint' || exponent < 0n) {
                throw new RangeError((0, errors_1.message)('sparseExponentsNonNegativeBigInt'));
            }
        }
    }
    /**
     * Compares exponent vectors using the three global monomial orders required by
     * the polynomial algorithms.
     */
    static compareMonomials(a, b, order) {
        if (order === 'lex') {
            for (let i = 0; i < a.length; i++) {
                if (a[i] !== b[i]) {
                    return a[i] > b[i] ? 1 : -1;
                }
            }
            return 0;
        }
        let degreeA = 0n;
        let degreeB = 0n;
        for (let i = 0; i < a.length; i++) {
            degreeA += a[i];
            degreeB += b[i];
        }
        if (degreeA !== degreeB) {
            return degreeA > degreeB ? 1 : -1;
        }
        if (order === 'grlex') {
            for (let i = 0; i < a.length; i++) {
                if (a[i] !== b[i]) {
                    return a[i] > b[i] ? 1 : -1;
                }
            }
            return 0;
        }
        for (let i = a.length - 1; i >= 0; i--) {
            if (a[i] !== b[i]) {
                return a[i] < b[i] ? 1 : -1;
            }
        }
        return 0;
    }
    /** Creates a constant polynomial in a ring with variableCount variables. */
    static constant(variableCount, coefficient) {
        return new SparsePolynomial(variableCount, [
            {
                coefficient,
                exponents: new Array(variableCount).fill(0n),
            },
        ]);
    }
    /**
     * Encodes an exponent vector only for map identity.
     *
     * The polynomial retains the exponent vector beside the key, so no decoder exists.
     */
    static keyOf(exponents) {
        return exponents.join(',');
    }
    /** Creates a one-term polynomial. */
    static monomial(variableCount, coefficient, exponents) {
        return new SparsePolynomial(variableCount, [{ coefficient, exponents }]);
    }
    /** Creates the variable at variableIndex with coefficient one. */
    static variable(variableCount, variableIndex) {
        if (!Number.isSafeInteger(variableIndex) ||
            variableIndex < 0 ||
            variableIndex >= variableCount) {
            throw new RangeError((0, errors_1.message)('sparseVariableIndexOutOfRing'));
        }
        const exponents = new Array(variableCount).fill(0n);
        exponents[variableIndex] = 1n;
        return new SparsePolynomial(variableCount, [{ coefficient: 1n, exponents }]);
    }
    /** Creates the zero polynomial in a ring with variableCount variables. */
    static zero(variableCount) {
        return new SparsePolynomial(variableCount);
    }
    /**
     * Adds another polynomial in the same ring.
     *
     * Like monomials are combined exactly and zero coefficients are removed from
     * the returned polynomial.
     */
    add(other) {
        if (this.variableCount !== other.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseAdditionRingMismatch'));
        }
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            result.termsByKey.set(key, term);
        }
        for (const [key, term] of other.termsByKey) {
            const existing = result.termsByKey.get(key);
            const coefficient = (existing?.coefficient ?? 0n) + term.coefficient;
            if (coefficient === 0n) {
                result.termsByKey.delete(key);
            }
            else if (existing === undefined || coefficient !== existing.coefficient) {
                result.termsByKey.set(key, Object.freeze({
                    coefficient,
                    exponents: existing?.exponents ?? term.exponents,
                }));
            }
        }
        return result;
    }
    /**
     * Returns the coefficient of one monomial.
     *
     * @param exponents - Exact exponent vector for the requested monomial.
     */
    coefficient(exponents) {
        SparsePolynomial.assertExponents(this.variableCount, exponents);
        return this.termsByKey.get(SparsePolynomial.keyOf(exponents))?.coefficient ?? 0n;
    }
    /**
     * Returns the coefficient polynomial at one exact power of a selected variable.
     *
     * The selected variable remains in the ring with exponent zero so recursive
     * algorithms can preserve the original coordinate layout.
     */
    coefficientIn(variableIndex, exponent) {
        this.degree(variableIndex);
        if (exponent < 0n) {
            throw new RangeError((0, errors_1.message)('sparseCoefficientExponentNonNegative'));
        }
        const terms = [];
        for (const term of this.termsByKey.values()) {
            if (term.exponents[variableIndex] !== exponent) {
                continue;
            }
            const exponents = [...term.exponents];
            exponents[variableIndex] = 0n;
            terms.push({
                coefficient: term.coefficient,
                exponents,
            });
        }
        return new SparsePolynomial(this.variableCount, terms);
    }
    /** Returns the constant coefficient, or zero when the constant monomial is absent. */
    constantTerm() {
        const exponents = new Array(this.variableCount).fill(0n);
        return this.termsByKey.get(SparsePolynomial.keyOf(exponents))?.coefficient ?? 0n;
    }
    /**
     * Returns the positive greatest common divisor of all coefficients.
     *
     * The zero polynomial has coefficient content zero.
     */
    content() {
        let content = 0n;
        for (const term of this.termsByKey.values()) {
            content = (0, bigint_1.GCD)(content, term.coefficient);
            if (content === 1n) {
                break;
            }
        }
        return content;
    }
    /**
     * Returns the degree in one variable.
     *
     * The zero polynomial has no degree and returns null.
     */
    degree(variableIndex) {
        if (!Number.isSafeInteger(variableIndex) ||
            variableIndex < 0 ||
            variableIndex >= this.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseVariableIndexOutOfRing'));
        }
        if (this.isZero()) {
            return null;
        }
        let degree = 0n;
        for (const term of this.termsByKey.values()) {
            if (term.exponents[variableIndex] > degree) {
                degree = term.exponents[variableIndex];
            }
        }
        return degree;
    }
    /**
     * Differentiates with respect to one ring variable.
     *
     * Coefficients and exponents stay in bigint arithmetic, including exponents
     * above JavaScript's safe-integer range.
     */
    derivative(variableIndex) {
        this.degree(variableIndex);
        const terms = [];
        for (const term of this.termsByKey.values()) {
            const exponent = term.exponents[variableIndex];
            if (exponent === 0n) {
                continue;
            }
            const exponents = term.exponents.map(value => value);
            exponents[variableIndex] = exponent - 1n;
            terms.push({
                coefficient: term.coefficient * exponent,
                exponents,
            });
        }
        return new SparsePolynomial(this.variableCount, terms);
    }
    /**
     * Divides every term by one monomial when the quotient stays in the integer
     * polynomial ring.
     *
     * Returns null when any coefficient or exponent is not exactly divisible.
     */
    divideByMonomial(divisor) {
        if (divisor.coefficient === 0n) {
            throw new RangeError((0, errors_1.message)('sparseZeroMonomialDivision'));
        }
        SparsePolynomial.assertExponents(this.variableCount, divisor.exponents);
        const terms = [];
        for (const term of this.termsByKey.values()) {
            if (term.coefficient % divisor.coefficient !== 0n) {
                return null;
            }
            const exponents = new Array(this.variableCount);
            for (let variableIndex = 0; variableIndex < this.variableCount; variableIndex++) {
                if (term.exponents[variableIndex] < divisor.exponents[variableIndex]) {
                    return null;
                }
                exponents[variableIndex] =
                    term.exponents[variableIndex] - divisor.exponents[variableIndex];
            }
            terms.push({
                coefficient: term.coefficient / divisor.coefficient,
                exponents,
            });
        }
        return new SparsePolynomial(this.variableCount, terms);
    }
    /**
     * Divides all coefficients by an integer scalar.
     *
     * @throws RangeError When the scalar is zero or does not divide every coefficient.
     */
    divideByScalarExact(divisor) {
        if (divisor === 0n) {
            throw new RangeError((0, errors_1.message)('sparseZeroScalarDivision'));
        }
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            if (term.coefficient % divisor !== 0n) {
                throw new RangeError((0, errors_1.message)('sparseScalarDivisionExact'));
            }
            result.termsByKey.set(key, Object.freeze({
                coefficient: term.coefficient / divisor,
                exponents: term.exponents,
            }));
        }
        return result;
    }
    /**
     * Divides by one polynomial using exact integer coefficient arithmetic.
     *
     * Returns null as soon as the current leading term is not exactly divisible
     * by the divisor leading term. A successful result therefore has zero
     * remainder in the selected monomial order.
     */
    divideExact(divisor, order = 'lex') {
        if (this.variableCount !== divisor.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseDivisionRingMismatch'));
        }
        const lead = divisor.leadingTerm(order);
        if (lead === null) {
            throw new RangeError((0, errors_1.message)('sparseZeroPolynomialDivision'));
        }
        if (this.isZero()) return SparsePolynomial.zero(this.variableCount);
        if (divisor.termCount === 1) return this.divideByMonomial(lead);
        // A genuine factor cannot increase the degree in any indeterminate.
        for (let i = 0; i < this.variableCount; ++i) {
            if (this.degree(i) < divisor.degree(i)) return null;
        }
        // Try a packed 16-bit monomial-key / IEEE-754 safe-integer division.
        // On an overflowing intermediate, restart using exact BigInt arithmetic.
        const fast = fastExactDivide(this, divisor, order);
        if (fast !== FAST_FALLBACK) return fast;
        const n = this.variableCount;
        const residual = new Map(this.termsByKey);
        const quotient = SparsePolynomial.zero(n);
        const ds = divisor.termsByKey.values();
        const divisorTerms = Array.from(ds);
        while (residual.size) {
            let leading = null;
            for (const term of residual.values()) {
                if (leading === null || SparsePolynomial.compareMonomials(term.exponents, leading.exponents, order) > 0) leading = term;
            }
            if (leading.coefficient % lead.coefficient !== 0n) return null;
            const qex = new Array(n);
            for (let i = 0; i < n; ++i) {
                if (leading.exponents[i] < lead.exponents[i]) return null;
                qex[i] = leading.exponents[i] - lead.exponents[i];
            }
            const qc = leading.coefficient / lead.coefficient;
            const qkey = SparsePolynomial.keyOf(qex);
            const oldq = quotient.termsByKey.get(qkey);
            quotient.termsByKey.set(qkey, Object.freeze({ coefficient: (oldq?.coefficient ?? 0n) + qc, exponents: oldq?.exponents ?? Object.freeze(qex) }));
            for (const term of divisorTerms) {
                const ex = term.exponents.map((e, i) => e + qex[i]);
                const key = SparsePolynomial.keyOf(ex);
                const old = residual.get(key);
                const coefficient = (old?.coefficient ?? 0n) - qc * term.coefficient;
                if (coefficient === 0n) residual.delete(key);
                else residual.set(key, {coefficient, exponents: old?.exponents ?? Object.freeze(ex)});
            }
        }
        return quotient;
    }
    /** Returns whether both polynomials have the same ring and exact nonzero terms. */
    equals(other) {
        if (this.variableCount !== other.variableCount || this.termCount !== other.termCount) {
            return false;
        }
        for (const [key, term] of this.termsByKey) {
            if (other.termsByKey.get(key)?.coefficient !== term.coefficient) {
                return false;
            }
        }
        return true;
    }
    /**
     * Substitutes one variable with an exact integer while retaining the same ring.
     *
     * Terms that become like terms after substitution are combined in the result.
     */
    evaluateVariable(variableIndex, value) {
        this.degree(variableIndex);
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            const exponent = term.exponents[variableIndex];
            let coordinateStart = 0;
            for (let index = 0; index < variableIndex; index++) {
                coordinateStart = key.indexOf(',', coordinateStart) + 1;
            }
            const separator = key.indexOf(',', coordinateStart);
            const coordinateEnd = separator === -1 ? key.length : separator;
            const evaluatedKey = key.slice(0, coordinateStart) + '0' + key.slice(coordinateEnd);
            const existing = result.termsByKey.get(evaluatedKey);
            const contribution = term.coefficient * value ** exponent;
            const coefficient = (existing?.coefficient ?? 0n) + contribution;
            if (coefficient === 0n) {
                result.termsByKey.delete(evaluatedKey);
            }
            else if (existing !== undefined) {
                if (coefficient !== existing.coefficient) {
                    result.termsByKey.set(evaluatedKey, Object.freeze({
                        coefficient,
                        exponents: existing.exponents,
                    }));
                }
            }
            else if (contribution !== 0n) {
                if (exponent === 0n) {
                    result.termsByKey.set(evaluatedKey, term);
                }
                else {
                    const exponents = [...term.exponents];
                    exponents[variableIndex] = 0n;
                    result.termsByKey.set(evaluatedKey, Object.freeze({
                        coefficient,
                        exponents: Object.freeze(exponents),
                    }));
                }
            }
        }
        return result;
    }
    /**
     * Substitutes several variables with exact integers in one sparse traversal.
     *
     * The selected coordinates remain in the polynomial ring with exponent zero.
     */
    evaluateVariables(variableIndices, values) {
        if (variableIndices.length !== values.length) {
            throw new RangeError((0, errors_1.message)('wrongInput', {
                expected: `${variableIndices.length} evaluation values`,
                received: `${values.length}`,
            }));
        }
        if (variableIndices.length === 0) {
            return this;
        }
        const selected = new Set();
        for (const variableIndex of variableIndices) {
            this.degree(variableIndex);
            if (selected.has(variableIndex)) {
                throw new RangeError((0, errors_1.message)('wrongInput', {
                    expected: 'unique evaluation variables',
                    received: 'duplicate variable index',
                }));
            }
            selected.add(variableIndex);
        }
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            let coefficient = term.coefficient;
            let changesMonomial = false;
            for (let index = 0; index < variableIndices.length; index++) {
                const variableIndex = variableIndices[index];
                const exponent = term.exponents[variableIndex];
                if (exponent === 0n) {
                    continue;
                }
                changesMonomial = true;
                const value = values[index];
                if (value === 0n) {
                    coefficient = 0n;
                    break;
                }
                if (value === 1n) {
                    continue;
                }
                if (value === -1n) {
                    if ((exponent & 1n) === 1n) {
                        coefficient = -coefficient;
                    }
                    continue;
                }
                coefficient *= value ** exponent;
            }
            if (coefficient === 0n) {
                continue;
            }
            let evaluatedKey = key;
            let exponents = term.exponents;
            if (changesMonomial) {
                const projected = [...term.exponents];
                for (const variableIndex of variableIndices) {
                    projected[variableIndex] = 0n;
                }
                exponents = Object.freeze(projected);
                evaluatedKey = SparsePolynomial.keyOf(exponents);
            }
            const existing = result.termsByKey.get(evaluatedKey);
            const combined = (existing?.coefficient ?? 0n) + coefficient;
            if (combined === 0n) {
                result.termsByKey.delete(evaluatedKey);
            }
            else if (existing === undefined) {
                result.termsByKey.set(evaluatedKey, changesMonomial || coefficient !== term.coefficient
                    ? Object.freeze({ coefficient, exponents })
                    : term);
            }
            else if (combined !== existing.coefficient) {
                result.termsByKey.set(evaluatedKey, Object.freeze({
                    coefficient: combined,
                    exponents: existing.exponents,
                }));
            }
        }
        return result;
    }
    /** Returns whether the polynomial is zero or contains only a constant term. */
    isConstant() {
        if (this.isZero()) {
            return true;
        }
        if (this.termsByKey.size !== 1) {
            return false;
        }
        const term = this.termsByKey.values().next().value;
        return term !== undefined && term.exponents.every(exponent => exponent === 0n);
    }
    /** Returns whether the polynomial has no nonzero terms. */
    isZero() {
        return this.termsByKey.size === 0;
    }
    /**
     * Returns the coefficient polynomial of the highest power of one variable.
     *
     * The selected coordinate remains present with exponent zero.
     */
    leadingCoefficientIn(variableIndex) {
        const degree = this.degree(variableIndex);
        return degree === null ? null : this.coefficientIn(variableIndex, degree);
    }
    /**
     * Returns the leading term under the requested monomial order.
     *
     * Variable priority follows exponent-vector index order.
     */
    leadingTerm(order = 'lex') {
        if (order !== 'lex' && order !== 'grlex' && order !== 'grevlex') {
            throw new RangeError((0, errors_1.message)('sparseMonomialOrderUnsupported'));
        }
        let leading = null;
        for (const term of this.termsByKey.values()) {
            if (leading === null ||
                SparsePolynomial.compareMonomials(term.exponents, leading.exponents, order) > 0) {
                leading = term;
            }
        }
        return leading;
    }
    /**
     * Applies an exact bigint transformation to every stored coefficient while
     * preserving the existing monomial keys and exponent vectors.
     */
    mapCoefficients(mapper) {
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            const coefficient = mapper(term.coefficient);
            if (coefficient === 0n) {
                continue;
            }
            result.termsByKey.set(key, coefficient === term.coefficient
                ? term
                : Object.freeze({
                    coefficient,
                    exponents: term.exponents,
                }));
        }
        return result;
    }
    /** Returns the largest absolute coefficient, or zero for the zero polynomial. */
    maxAbsoluteCoefficient() {
        let maximum = 0n;
        for (const term of this.termsByKey.values()) {
            const absolute = term.coefficient < 0n ? -term.coefficient : term.coefficient;
            if (absolute > maximum) {
                maximum = absolute;
            }
        }
        return maximum;
    }
    /**
     * Computes the difference of two monomial-scaled polynomials in one sparse
     * traversal. This avoids constructing the monomial factors and multiplication
     * results used by fraction-free S-polynomials.
     */
    monomialScaleSubtract(leftScalar, leftMonomialExponents, other, rightScalar, rightMonomialExponents) {
        if (this.variableCount !== other.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseSubtractionRingMismatch'));
        }
        SparsePolynomial.assertExponents(this.variableCount, leftMonomialExponents);
        SparsePolynomial.assertExponents(this.variableCount, rightMonomialExponents);
        const result = SparsePolynomial.zero(this.variableCount);
        const accumulate = (polynomial, scalar, monomialExponents) => {
            if (scalar === 0n) {
                return;
            }
            for (const term of polynomial.termsByKey.values()) {
                const exponents = new Array(this.variableCount);
                for (let variableIndex = 0; variableIndex < this.variableCount; variableIndex++) {
                    exponents[variableIndex] =
                        term.exponents[variableIndex] + monomialExponents[variableIndex];
                }
                const frozenExponents = Object.freeze(exponents);
                const key = SparsePolynomial.keyOf(frozenExponents);
                const existing = result.termsByKey.get(key);
                const coefficient = (existing?.coefficient ?? 0n) + term.coefficient * scalar;
                if (coefficient === 0n) {
                    result.termsByKey.delete(key);
                }
                else {
                    result.termsByKey.set(key, Object.freeze({ coefficient, exponents: existing?.exponents ?? frozenExponents }));
                }
            }
        };
        accumulate(this, leftScalar, leftMonomialExponents);
        accumulate(other, -rightScalar, rightMonomialExponents);
        return result;
    }
    /**
     * Multiplies by another polynomial in the same ring.
     *
     * Monomial exponents are added as bigints. The constructor combines products
     * that land on the same monomial and removes coefficients that cancel to zero.
     */
    multiply(other) {
        if (this.variableCount !== other.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseMultiplicationRingMismatch'));
        }
        if (this.isZero() || other.isZero()) return SparsePolynomial.zero(this.variableCount);
        // Small results usually cost less in pure BigInt than in conversion.
        if (this.termCount * other.termCount >= 16) {
            const fast = fastExactMultiply(this, other);
            if (fast !== FAST_FALLBACK) return fast;
        }
        // Accumulate directly into the output map.  In particular, do not
        // materialize O(n*m) duplicate term objects in the constructor.
        const result = SparsePolynomial.zero(this.variableCount);
        const count = this.variableCount;
        for (const left of this.termsByKey.values()) {
            for (const right of other.termsByKey.values()) {
                const ex = new Array(count);
                for (let i = 0; i < count; ++i) ex[i] = left.exponents[i] + right.exponents[i];
                const key = SparsePolynomial.keyOf(ex);
                const existing = result.termsByKey.get(key);
                const coefficient = (existing?.coefficient ?? 0n) + left.coefficient * right.coefficient;
                if (coefficient === 0n) result.termsByKey.delete(key);
                else result.termsByKey.set(key, Object.freeze({coefficient, exponents: existing?.exponents ?? Object.freeze(ex)}));
            }
        }
        return result;
    }
    /**
     * Returns the additive inverse of this polynomial.
     */
    negate() {
        return this.scale(-1n);
    }
    /**
     * Makes the leading coefficient positive without changing coefficient content.
     */
    normalizeLeadingSign(order = 'lex') {
        const leading = this.leadingTerm(order);
        return leading !== null && leading.coefficient < 0n ? this.negate() : this;
    }
    /** Raises the polynomial to a non-negative exact integer power. */
    pow(exponent) {
        if (exponent < 0n) {
            throw new RangeError((0, errors_1.message)('sparseExponentNonNegative'));
        }
        let result = SparsePolynomial.constant(this.variableCount, 1n);
        let power = exponent;
        if (power === 0n) {
            return result;
        }
        if ((power & 1n) === 1n) {
            result = result.multiply(this);
        }
        power >>= 1n;
        if (power === 0n) {
            return result;
        }
        let factor = this.multiply(this);
        while (power > 0n) {
            if ((power & 1n) === 1n) {
                result = result.multiply(factor);
            }
            power >>= 1n;
            if (power > 0n) {
                factor = factor.multiply(factor);
            }
        }
        return result;
    }
    /**
     * Divides by positive coefficient content while preserving the polynomial sign.
     */
    primitivePart() {
        const content = this.content();
        return content === 0n || content === 1n ? this : this.divideByScalarExact(content);
    }
    /**
     * Multiplies every coefficient by an exact integer scalar.
     */
    scale(scalar) {
        if (scalar === 0n || this.isZero()) {
            return SparsePolynomial.zero(this.variableCount);
        }
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            result.termsByKey.set(key, Object.freeze({
                coefficient: term.coefficient * scalar,
                exponents: term.exponents,
            }));
        }
        return result;
    }
    /**
     * Computes a scaled subtraction where the right polynomial is also multiplied
     * by one monomial. This avoids constructing the two scaled intermediate
     * polynomials used by fraction-free reduction.
     */
    scaleSubtractMonomial(leftScalar, other, rightScalar, monomialExponents) {
        if (this.variableCount !== other.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseSubtractionRingMismatch'));
        }
        SparsePolynomial.assertExponents(this.variableCount, monomialExponents);
        const result = SparsePolynomial.zero(this.variableCount);
        if (leftScalar !== 0n) {
            for (const [key, term] of this.termsByKey) {
                const coefficient = term.coefficient * leftScalar;
                if (coefficient !== 0n) {
                    result.termsByKey.set(key, Object.freeze({ coefficient, exponents: term.exponents }));
                }
            }
        }
        if (rightScalar === 0n) {
            return result;
        }
        for (const term of other.termsByKey.values()) {
            const exponents = new Array(this.variableCount);
            for (let variableIndex = 0; variableIndex < this.variableCount; variableIndex++) {
                exponents[variableIndex] =
                    term.exponents[variableIndex] + monomialExponents[variableIndex];
            }
            const frozenExponents = Object.freeze(exponents);
            const key = SparsePolynomial.keyOf(frozenExponents);
            const existing = result.termsByKey.get(key);
            const coefficient = (existing?.coefficient ?? 0n) - term.coefficient * rightScalar;
            if (coefficient === 0n) {
                result.termsByKey.delete(key);
            }
            else {
                result.termsByKey.set(key, Object.freeze({
                    coefficient,
                    exponents: existing?.exponents ?? frozenExponents,
                }));
            }
        }
        return result;
    }
    /**
     * Subtracts another polynomial in the same ring.
     */
    subtract(other) {
        if (this.variableCount !== other.variableCount) {
            throw new RangeError((0, errors_1.message)('sparseSubtractionRingMismatch'));
        }
        const result = SparsePolynomial.zero(this.variableCount);
        for (const [key, term] of this.termsByKey) {
            result.termsByKey.set(key, term);
        }
        for (const [key, term] of other.termsByKey) {
            const existing = result.termsByKey.get(key);
            const coefficient = (existing?.coefficient ?? 0n) - term.coefficient;
            if (coefficient === 0n) {
                result.termsByKey.delete(key);
            }
            else if (existing === undefined || coefficient !== existing.coefficient) {
                result.termsByKey.set(key, Object.freeze({
                    coefficient,
                    exponents: existing?.exponents ?? term.exponents,
                }));
            }
        }
        return result;
    }
    /** Number of stored nonzero terms. */
    get termCount() {
        return this.termsByKey.size;
    }
    /**
     * Returns a copy of the stored terms.
     *
     * The exponent arrays themselves are frozen so callers cannot alter polynomial storage.
     */
    terms() {
        return Array.from(this.termsByKey.values());
    }
    /**
     * Returns the largest total degree among the stored monomials.
     *
     * The zero polynomial has no total degree and returns null.
     */
    totalDegree() {
        if (this.isZero()) {
            return null;
        }
        let maxDegree = 0n;
        for (const term of this.termsByKey.values()) {
            let degree = 0n;
            for (const exponent of term.exponents) {
                degree += exponent;
            }
            if (degree > maxDegree) {
                maxDegree = degree;
            }
        }
        return maxDegree;
    }
    /** Returns the variable indices that occur with a positive exponent. */
    variables() {
        const present = new Set();
        for (const term of this.termsByKey.values()) {
            for (let variableIndex = 0; variableIndex < this.variableCount; variableIndex++) {
                if (term.exponents[variableIndex] !== 0n) {
                    present.add(variableIndex);
                }
            }
        }
        return Array.from(present).sort((a, b) => a - b);
    }
}
exports.SparsePolynomial = SparsePolynomial;

};
modules["core/functions/bigint/bigint"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.factorial = void 0;
exports.abs = abs;
exports.uniqueSortedArray = uniqueSortedArray;
exports.GCD = GCD;
exports.LCM = LCM;
exports.simplifyRatio = simplifyRatio;
exports.convert = convert;
exports.sign = sign;
exports.isEven = isEven;
exports.integerSquareRoot = integerSquareRoot;
exports.ceilIntegerSquareRoot = ceilIntegerSquareRoot;
exports.fibonacci = fibonacci;
exports.mod = mod;
exports.powMod = powMod;
exports.invMod = invMod;
exports.productInRange = productInRange;
exports.max = max;
exports.min = min;
exports.recoverCoeffsZ = recoverCoeffsZ;
exports.binom = binom;
exports.bigintLCM = bigintLCM;
const errors_1 = require("../../errors");
const string_1 = require("../string");
const utils_1 = require("../utils");
/**
 * Returns the absolute value of a bigint
 *
 * @param x
 * @returns
 */
function abs(x) {
    return x < 0n ? -x : x;
}
/**
 * Returns an array with unique values, sorted, and in reverse order.
 *
 * @param array
 * @returns
 */
function uniqueSortedArray(array) {
    return (0, utils_1.arrayUnique)([].slice.call(array).map(function (x) {
        return abs(x);
    })).sort();
}
/**
 * Calculates the gcd of a set of bigints
 *
 * @returns
 */
function GCD(...args) {
    // Get a unique set and convert to absolute values
    args = uniqueSortedArray(args);
    // Point to the first element
    let a = args[0];
    for (let i = 1; i < args.length; i++) {
        let b = args[i];
        while (true) {
        casBudgetCheck();
            a %= b;
            if (a === 0n) {
                a = b;
                break;
            }
            b %= a;
            if (b === 0n) {
                break;
            }
        }
    }
    return a;
}
/**
 * Calculates the LCM of a set of bigints
 *
 * @param args
 */
function LCM(...args) {
    // Get a unique set and convert to absolute values
    args = uniqueSortedArray(args);
    let a = args[0];
    for (let i = 1; i < args.length; i++) {
        const b = args[i];
        // Get the LCM of that of the last calculated LCM and the new number
        a = (a * b) / GCD(a, b);
    }
    return a;
}
/**
 * Simplifies a ratio of two bigints
 *
 * @param numerator
 * @param denominator
 * @returns
 */
function simplifyRatio(numerator, denominator) {
    if (denominator !== 1n) {
        const gcd = GCD(denominator, numerator);
        return [numerator / gcd, denominator / gcd];
    }
    return [numerator, denominator];
}
/**
 * Converts a decimal number to a rational
 *
 * @param numStr
 * @returns
 */
function convert(numStr) {
    //Split the whole and decimal
    const [w, d] = (0, string_1.decimalSplit)(numStr);
    //Create the denominator based on the number of decimal places
    const denominator = 10n ** BigInt(d.length);
    const numerator = BigInt(w + d);
    return simplifyRatio(numerator, denominator);
}
/**
 * Returns the sign of a bigint
 *
 * @param n
 * @returns -1 if negative and 1 for positive and 0 for zero
 */
function sign(n) {
    return n === 0n ? 0 : n < 0n ? -1 : 1;
}
/**
 * Checks if a bigint is an even number
 *
 * @param n
 * @returns
 */
function isEven(n) {
    return n % 2n === 0n;
}
/**
 * Returns the floor of the exact square root of a non-negative bigint.
 */
function integerSquareRoot(value) {
    if (value < 0n) {
        throw new RangeError((0, errors_1.message)('integerSqrtNonnegative'));
    }
    if (value < 2n) {
        return value;
    }
    let root = 1n << ((BigInt(value.toString(2).length) + 1n) >> 1n);
    while (true) {
        casBudgetCheck();
        const next = (root + value / root) >> 1n;
        if (next >= root) {
            return root;
        }
        root = next;
    }
}
/** Returns the least bigint whose square is at least value. */
function ceilIntegerSquareRoot(value) {
    const root = integerSquareRoot(value);
    return root * root === value ? root : root + 1n;
}
/**
 * Computes the nth Fibonacci number exactly using fast doubling.
 *
 * Negative indices use the negafibonacci identity F(-n) = (-1)^(n+1)F(n).
 *
 * @param n - Integer index.
 * @returns The exact Fibonacci number.
 */
function fibonacci(n) {
    const isNegative = n < 0n;
    const index = isNegative ? -n : n;
    let a = 0n;
    let b = 1n;
    for (const bit of index.toString(2)) {
        const c = a * (2n * b - a);
        const d = a * a + b * b;
        if (bit === '0') {
            a = c;
            b = d;
        }
        else {
            a = d;
            b = c + d;
        }
    }
    let retval = a;
    if (isNegative && isEven(index)) {
        retval = -retval;
    }
    return retval;
}
/**
 * The big number version
 */
exports.factorial = (function () {
    const factorialCache = [1n, 1n];
    let i = 2;
    /**
     * Leverages JavaScript's native bigInt and the solution provide here:
     * https://stackoverflow.com/questions/3959211/what-is-the-fastest-factorial-function-in-javascript
     * to provide a fast factorial function capable of calculating large numbers.
     *
     * @remarks
     *
     * This is limited to Number.MAX_SAFE_INTEGER since n is of type number since for loop requires it
     *
     * @param n - The factorial of the number being calculated
     * @returns The computed factorial
     * @throws RangeError If `n` is negative.
     */
    function bigFactorial(n) {
        if (typeof n === 'bigint') {
            n = Number(n);
        }
        if (n < 0) {
            throw new RangeError((0, errors_1.message)('factorialNegativeUndefined'));
        }
        // If there's a cached version return that instead
        if (typeof factorialCache[n] != 'undefined') {
            return factorialCache[n];
        }
        let result = factorialCache[i - 1];
        for (; i <= n; i++) {
            factorialCache[i] = result = result * BigInt(i);
        }
        return result;
    }
    bigFactorial(100);
    return bigFactorial;
})();
// Cache the first 100
/**
 * Calculates the modulo of two numbers
 *
 * @param a
 * @param b
 * @returns
 */
function mod(a, b) {
    const signA = sign(a);
    const signB = sign(b);
    let retval;
    if (signA === signB) {
        retval = a % b;
    }
    else {
        // https://stackoverflow.com/questions/4467539/javascript-modulo-gives-a-negative-result-for-negative-numbers
        retval = ((a % b) + b) % b;
    }
    return retval;
}
/**
 * Raises a bigint base to a non-negative bigint exponent modulo a positive modulus.
 *
 * Binary exponentiation keeps every intermediate value in bigint modular arithmetic.
 */
function powMod(base, exponent, modulus) {
    if (modulus <= 0n) {
        throw new RangeError((0, errors_1.message)('modularExponentiationPositiveModulus'));
    }
    if (exponent < 0n) {
        throw new RangeError((0, errors_1.message)('modularExponentiationNonnegativeExponent'));
    }
    let result = 1n % modulus;
    let factor = mod(base, modulus);
    let power = exponent;
    while (power > 0n) {
        if (power % 2n === 1n) {
            result = (result * factor) % modulus;
        }
        factor = (factor * factor) % modulus;
        power /= 2n;
    }
    return result;
}
/**
 * Finds the multiplicative inverse of a modulo m.
 *
 * Returns -1 when the inverse does not exist. The extended Euclidean algorithm keeps
 * the operation exact for bigint inputs without scanning every possible residue.
 *
 * @param a
 * @param m
 * @returns
 */
function invMod(a, m) {
    let retval = -1n;
    const modulus = abs(m);
    if (modulus !== 0n) {
        let t = 0n;
        let nextT = 1n;
        let r = modulus;
        let nextR = mod(a, modulus);
        while (nextR !== 0n) {
            const q = r / nextR;
            [t, nextT] = [nextT, t - q * nextT];
            [r, nextR] = [nextR, r - q * nextR];
        }
        if (r === 1n) {
            retval = mod(t, modulus);
        }
    }
    return retval;
}
/**
 * Gets the product given a number range start and finish
 *
 * @param start The start of the range
 * @param end The end of the range
 * @param step The increment step
 */
function productInRange(start, end, step = 1) {
    let retval = BigInt(start);
    let n = BigInt(start);
    const i = BigInt(step);
    while (n < end) {
        n += i;
        retval *= n;
    }
    return retval;
}
/**
 * Returns the largest of a set of bigints
 * @param args
 * @returns
 */
function max(...args) {
    if (args.length < 1) {
        // Math.max() for Number returns -Infinity for empty lists,
        // but BigInt can't represent Infinity, so throwing an error or
        // handling as per application logic is necessary.
        throw new Error((0, errors_1.message)('maxBigIntRequiresArgument'));
    }
    let max = args[0];
    for (let i = 1; i < args.length; i++) {
        if (args[i] > max) {
            max = args[i];
        }
    }
    return max;
}
/**
 * Returns the smallest of a set of bigints
 * @param args
 * @returns
 */
function min(...args) {
    if (args.length === 0) {
        // Math.min() returns Infinity for no arguments, but BigInt cannot represent Infinity.
        // It's often better to throw an error or handle the empty case explicitly.
        throw new Error((0, errors_1.message)('minBigIntEmpty'));
    }
    let min = args[0];
    for (let i = 1; i < args.length; i++) {
        if (args[i] < min) {
            min = args[i];
        }
    }
    return min;
}
function recoverCoeffsZ(num, primes) {
    const variables = Object.keys(primes).sort((a, b) => {
        if (/\w+[-+]\w+/.test(a)) {
            return 1;
        }
        return a.localeCompare(b);
    });
    const map = {};
    // Loop though the variables
    for (const v of variables) {
        const prime = primes[v];
        let n = 0;
        let canDivide = true;
        // Divide out the max until you hit a remainder
        do {
            const q = num / prime;
            if (q === 0n) {
                break;
            }
            const r = num % q;
            if (r === 0n) {
                n++;
                num = q;
            }
            else {
                canDivide = false;
            }
        } while (canDivide);
        // Mark the power of the variable found
        map[v] = n;
    }
    // Put the rest to the coefficient
    map['coeff'] = num;
    return map;
}
/**
 * The binomial function using bigint
 * @param n
 * @param k
 * @returns
 */
function binom(n, k) {
    if (n < 0n) {
        throw new RangeError((0, errors_1.message)('binomBigIntNonnegativeN'));
    }
    if (k < 0n || k > n) {
        return 0n;
    }
    if (k === 0n || k === n) {
        return 1n;
    }
    let kk = k;
    const nMinusK = n - k;
    if (nMinusK < kk) {
        kk = nMinusK;
    }
    let result = 1n;
    for (let i = 1n; i <= kk; i++) {
        result = (result * (n - kk + i)) / i;
    }
    return result;
}
/**
 * Computes the LCM of an array of positive bigints.
 */
function bigintLCM(values) {
    if (values.length === 0) {
        return 1n;
    }
    function gcd(a, b) {
        a = a < 0n ? -a : a;
        b = b < 0n ? -b : b;
        while (b !== 0n) {
            [a, b] = [b, a % b];
        }
        return a;
    }
    function lcm(a, b) {
        return (a / gcd(a, b)) * b;
    }
    let result = values[0];
    for (let i = 1; i < values.length; i++) {
        result = lcm(result, values[i]);
    }
    return result;
}

};
modules["core/functions/utils"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.arrayUnique = arrayUnique;
exports.arrayAddUnique = arrayAddUnique;
exports.checkDecimal = checkDecimal;
exports.isSorted = isSorted;
/**
 * Removes duplicates from an array
 *
 * @param arr
 * @returns
 */
function arrayUnique(arr) {
    return [...new Set(arr)];
}
/**
 * Similar to concat but only adds elements if they're unique
 *
 * @param fromArray The array being added from
 * @param toArray The array being added to
 */
function arrayAddUnique(fromArray, toArray) {
    for (const x of fromArray) {
        if (!toArray.includes(x)) {
            toArray.push(x);
        }
    }
    return toArray;
}
/**
 * Checks to see if the inverse of decimal is within a max bound.
 * For example: given the number 0.01 it will return true for 0 - 100
 *
 * @param n
 * @param max
 * @returns
 */
function checkDecimal(n, max) {
    return 1 / (n % 1) <= max;
}
function isSorted(arr) {
    if (arr.length <= 1) {
        return true; // An empty array or an array with one element is considered sorted
    }
    for (let i = 1; i < arr.length; i++) {
        if (arr[i - 1] > arr[i]) {
            return false; // If any element is smaller than the previous one, the array is not sorted
        }
    }
    return true; // If the loop completes without returning false, the array is sorted
}

};
modules["core/functions/string"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.scientificToDecimal = scientificToDecimal;
exports.decimalSplit = decimalSplit;
exports.format = format;
const errors_1 = require("../errors");
/**
 * Removes the minus sign from the beginning of the string
 *
 * @param str
 * @returns an array with the first item as true if a minus
 * was found and the string minus the minus sign.
 */
function stripSign(str) {
    // Check if it has a minus sign
    const hasMinus = str.charAt(0) === '-';
    // Remove it if it does
    if (hasMinus || str.charAt(0) === '+') {
        str = str.substring(1);
    }
    return [hasMinus, str];
}
/**
 * Converts a string from scientific notation form to decimal form
 *
 * @param str
 * @returns
 */
function scientificToDecimal(str) {
    let isNegative;
    // Remove the sign by slicing the string
    [isNegative, str] = stripSign(str);
    // Split it into coefficient and exponent
    const [c, e] = str.toLowerCase().split('e');
    // Split the coefficient into the whole and decimal portion
    // eslint-disable-next-line prefer-const
    let [w, d] = c.split('.');
    // Provide and empty sting for safety if in the form n(e)n
    d = d || '';
    // The total length of the string
    const length = w.length + (d ? d.length : 0);
    // The total string minus the dot
    const numString = w + d;
    // If it's small then we need to calculate the leading zeros
    // The shift of the decimal place to the left
    const dotLocation = w.length + Number(e);
    // Is the dot needed or not
    const dot = dotLocation === length ? '' : '.';
    let value;
    if (dotLocation <= 0) {
        // Join the value but pad after the dot with zeroes
        value = `0${dot}${'0'.repeat(Math.abs(dotLocation))}${numString}`;
    }
    else if (dotLocation > length) {
        value = `${numString}${'0'.repeat(Math.abs(dotLocation - length))}`;
    }
    else {
        value = `${numString.substring(0, dotLocation)}${dot}${numString.substring(dotLocation)}`;
    }
    return isNegative ? '-' + value : value;
}
/**
 * Splits at the decimal but ensures that array length is always 2
 *
 * @param str
 * @returns
 */
function decimalSplit(str) {
    const parts = str.split('.');
    if (parts.length > 2) {
        throw new SyntaxError((0, errors_1.message)('invalidDecimalValue', { value: str }));
    }
    if (parts.length === 1) {
        parts.push('');
    }
    return parts;
}
/**
 * A helper function to replace multiple occurrences in a string. Tag the replacement
 * with a double curly.
 * @example format('Hi. My name is {{name}}.', {name: 'John Doe'})
 *
 * @param str
 * @param replacements
 * @returns
 */
function format(str, replacements) {
    const replace = function (str, replacement, newValue) {
        return str.replace(new RegExp(`{{${replacement}}}`, 'g'), newValue);
    };
    if (Array.isArray(replacements)) {
        for (let i = 0; i < replacements.length; i++) {
            str = replace(str, String(i), replacements[i]);
        }
    }
    else {
        for (const x in replacements) {
            str = replace(str, x, replacements[x]);
        }
    }
    return str;
}

};
modules["core/errors"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ErrorMessages = exports.EnglishErrorMessages = exports.AssignmentError = exports.ZeroToZeroPowerError = exports.MissingReferenceError = exports.NotImplementedError = exports.DimensionError = exports.NaNError = exports.MathError = exports.ParserSyntaxError = exports.UnsupportedOperationError = exports.PolynomialError = exports.UndefinedError = exports.NullError = exports.ParserError = exports.OperatorError = exports.DivisionByZeroError = exports.UnexpectedInputError = exports.UnexpectedDataType = exports.UnexpectedTokenError = void 0;
exports.registerLanguage = registerLanguage;
exports.message = message;
const string_1 = require("./functions/string");
const Settings_1 = require("./Settings");
/** Thrown when tokenization encounters invalid adjacency, punctuation, or bracket structure. */
class UnexpectedTokenError extends Error {
}
exports.UnexpectedTokenError = UnexpectedTokenError;
/** Thrown when an operation receives a Nerdamer entity of the wrong runtime type. */
class UnexpectedDataType extends Error {
}
exports.UnexpectedDataType = UnexpectedDataType;
/** Used when input parses successfully but does not have the form an operation requires. */
class UnexpectedInputError extends Error {
}
exports.UnexpectedInputError = UnexpectedInputError;
/** Thrown for exact division by zero. Small nonzero numerical values are not treated as zero. */
class DivisionByZeroError extends Error {
}
exports.DivisionByZeroError = DivisionByZeroError;
/** Used for invalid parser-operator definitions and operator dispatch failures. */
class OperatorError extends Error {
}
exports.OperatorError = OperatorError;
/** General parser error for construction, evaluation, and registered-function failures. */
class ParserError extends Error {
}
exports.ParserError = ParserError;
/** Thrown when evaluation requires a value but a construct produces no value. */
class NullError extends Error {
}
exports.NullError = NullError;
/** Used when a mathematical operation has no defined value in the handled domain. */
class UndefinedError extends Error {
}
exports.UndefinedError = UndefinedError;
/** Thrown when input cannot be represented as a polynomial under the requested requirements. */
class PolynomialError extends Error {
}
exports.PolynomialError = PolynomialError;
/** Thrown when an operation is known but unsupported for the supplied form or domain. */
class UnsupportedOperationError extends Error {
}
exports.UnsupportedOperationError = UnsupportedOperationError;
/** Parser syntax error kept separate from lower-level tokenization failures. */
class ParserSyntaxError extends Error {
}
exports.ParserSyntaxError = ParserSyntaxError;
/** General mathematical precondition error for cases without a more specific error type. */
class MathError extends Error {
}
exports.MathError = MathError;
/** Thrown when a required numeric conversion produces `NaN`. */
class NaNError extends Error {
}
exports.NaNError = NaNError;
/** Thrown when vector, matrix, or structured-entity dimensions are incompatible. */
class DimensionError extends Error {
}
exports.DimensionError = DimensionError;
/** Used for recognized operations that have not been implemented yet. */
class NotImplementedError extends Error {
}
exports.NotImplementedError = NotImplementedError;
/** Thrown when a named converter or pattern reference cannot be resolved. */
class MissingReferenceError extends Error {
}
exports.MissingReferenceError = MissingReferenceError;
/** Thrown for the indeterminate symbolic form `0^0`. */
class ZeroToZeroPowerError extends Error {
}
exports.ZeroToZeroPowerError = ZeroToZeroPowerError;
/** Thrown when code attempts to assign to a reserved or otherwise non-assignable parser name. */
class AssignmentError extends Error {
}
exports.AssignmentError = AssignmentError;
/** Internal localized message templates used when constructing Nerdamer errors. */
exports.EnglishErrorMessages = {
    // General
    notImplemented: 'This method has not yet been implemented.',
    noInverse: 'Unable to calculate inverse!',
    undefinedValue: 'The requested value is undefined!',
    nullValue: 'Value expected but received null.',
    univariateInputOnly: 'The input for this function must be univariate!',
    differentSignsAtEndPointsRequired: 'Function must have opposite signs at endpoints a and b!',
    integerRequired: 'This function requires the coefficient to be an integer!',
    plainVariableExpected: 'Expected plain variable! Received {{input}}.',
    missingReference: 'The reference "{{ref}}" does not exist on this pattern!!',
    undefinedDivision: 'Division not defined expression of this type! {{type}}',
    restrictedVariableName: 'Cannot assign the requested value to the reserved name "{{name}}"!',
    noVariableInExpression: 'No variable found in expression!',
    quadraticExpressionExpected: 'Expression must be quadratic!',
    wrongInput: 'Wrong input! Expected {{expected}} but received {{received}}.',
    unsupportedFunction: 'Unsupported function {{function}}',
    functionMinArgs: '{{function}} requires a minimum of {{min}} arguments but {{actual}} provided!',
    functionMaxArgs: '{{function}} allows a maximum of {{max}} arguments but {{actual}} provided!',
    invalidFunctionDeclaration: 'Invalid function declaration {{declaration}}.',
    indexOutOfBounds: 'Index {{index}} out of bounds for {{type}} of length {{length}}',
    matrixRowIndexOutOfBounds: 'Row index {{index}} out of bounds for Matrix with {{length}} rows',
    matrixColumnIndexOutOfBounds: 'Column index {{index}} out of bounds for Matrix with {{length}} columns',
    dictionaryKeyNotFound: 'Key "{{key}}" not found in Dictionary. Available keys: {{keys}}',
    rangeValuesFinite: 'Range values must be finite.',
    rangeStepNonzero: 'Step cannot be zero',
    rangeStepNoAdvance: 'Step does not advance range at current precision',
    sparseVariableCountInvalid: 'SparsePolynomial variableCount must be a non-negative safe integer.',
    sparseExponentVectorLengthMismatch: 'SparsePolynomial exponent vector length must equal variableCount.',
    sparseExponentsNonNegativeBigInt: 'SparsePolynomial exponents must be non-negative bigints.',
    sparseVariableIndexOutOfRing: 'SparsePolynomial variable index is outside the polynomial ring.',
    sparseAdditionRingMismatch: 'SparsePolynomial addition requires the same polynomial ring.',
    sparseCoefficientExponentNonNegative: 'SparsePolynomial coefficient exponent must be non-negative.',
    sparseZeroMonomialDivision: 'SparsePolynomial cannot divide by a zero monomial.',
    sparseZeroScalarDivision: 'SparsePolynomial cannot divide by zero.',
    sparseScalarDivisionExact: 'SparsePolynomial scalar division must be exact for every coefficient.',
    sparseDivisionRingMismatch: 'SparsePolynomial division requires the same polynomial ring.',
    sparseZeroPolynomialDivision: 'SparsePolynomial cannot divide by the zero polynomial.',
    sparseMonomialOrderUnsupported: 'SparsePolynomial monomial order is not supported.',
    sparseMultiplicationRingMismatch: 'SparsePolynomial multiplication requires the same polynomial ring.',
    sparseExponentNonNegative: 'SparsePolynomial exponent must be non-negative.',
    sparseSubtractionRingMismatch: 'SparsePolynomial subtraction requires the same polynomial ring.',
    invalidDecimal: 'Invalid Decimal: "{{value}}"',
    invalidInterval: 'Invalid interval: ({{start}}, {{end}})',
    invalidAssumption: 'Invalid assumption: "{{input}}"',
    assumptionIdentifierExpected: 'Expected identifier',
    assumptionOperatorMissing: 'Missing operator',
    assumptionValueMissing: 'Missing value',
    assumptionOperatorUnsupported: 'Unsupported operator "{{operator}}"',
    matrixStringKeyUnsupported: 'Matrix does not support string key access',
    matrixTraceUnavailable: 'Unable to calculate trace for the matrix',
    defintMaxEvals: 'defint: maxEvals exceeded ({{max}}).',
    defintNonFiniteValue: 'defint: non-finite f(x) at x={{x}}',
    defintEndpointSingularity: 'defint: endpoint singularity could not be regularized near x={{x}}',
    defintSingularity: 'defint: singular/non-finite encountered near [{{a}}, {{b}}]',
    defintInfiniteBounds: 'defint: infinite bounds require allowInfinite: true',
    defintUnsupportedInfiniteBounds: 'defint: unsupported infinite bound configuration',
    symbolicAccessNamedTargetRequired: 'Symbolic indexed access requires a named Vector or Matrix target.',
    symbolicVectorIndexCount: 'Symbolic Vector access requires exactly one index.',
    symbolicMatrixRowNonScalar: 'Symbolic Matrix row access is not scalar and cannot be deferred as an Expression.',
    symbolicMatrixCellIndexCount: 'Symbolic Matrix cell access requires exactly two indices.',
    symbolicAccessMaxIndices: 'Symbolic indexed access supports at most two unresolved indices.',
    symbolicAccessIndexScalar: 'Symbolic accessor indices must evaluate to scalar expressions.',
    symbolicAccessTargetType: 'Symbolic indexed access target must resolve to a Vector or Matrix.',
    expectedOperatorOrFunction: 'Expected operator or function name but "{{value}}" found!',
    missingOpeningBracket: 'Missing opening bracket for "{{bracket}}":{{position}}',
    missingClosingBracket: 'Missing closing bracket for "{{bracket}}":{{position}}',
    prefixOperatorExpected: 'Prefix operator expected but {{operator}} encountered.',
    groebnerBasisSizeBudgetExceeded: 'Groebner budget exceeded: basis size > {{max}}',
    groebnerPairsBudgetExceeded: 'Groebner budget exceeded: pairsPopped > {{max}}',
    unsupportedParserEntity: 'Unsupported parser entity',
    factorialNegativeUndefined: 'Factorial is undefined for negative integers.',
    erfConvergenceFailed: 'erf not converging. Exiting!',
    defintInfiniteBoundsDecimal: 'defint: infinite bounds require allowInfinite:true',
    seqStartsAtInteger: "startsAt must be an integer",
    seqPolynomialOrMaxPower: "Sequence is not a polynomial or exceeds maxPower",
    seqInsufficientData: "Insufficient data points for detected degree",
    arithPowNegativeExponent: "pow: negative exponent",
    arithPowNNonnegativeInteger: "powN: exponent must be a nonnegative integer",
    arithBigIntSqrtNegative: "bigIntSqrt: negative input",
    arithNthRootPositiveInteger: "intNthRootExact: n must be a positive integer",
    arithDivisorsPositive: "getDivisors: n must be positive",
    arithModNormNonzeroModulus: "modNorm: modulus must be nonzero",
    arithModSymmetricNonzeroModulus: "modSymmetric: modulus must be nonzero",
    arithModInvZeroNoninvertible: "modInv: non-invertible (0)",
    arithModInvNoninvertible: "modInv: non-invertible",
    integerSqrtNonnegative: "Integer square root requires a non-negative value.",
    modularExponentiationPositiveModulus: "Modular exponentiation requires a positive modulus.",
    modularExponentiationNonnegativeExponent: "Modular exponentiation requires a non-negative exponent.",
    maxBigIntRequiresArgument: "maxBigInt() requires at least one argument",
    minBigIntEmpty: "Min of empty list is not defined for BigInt.",
    binomBigIntNonnegativeN: "binomBigInt: n must be non-negative",
    invalidDecimalValue: "Invalid decimal value: {{value}}",
    subtractionNotDefined: "Subtraction not defined for {{leftType}}-{{rightType}} | {{left}}-{{right}}",
    sparseAdapterVariablesUnique: "Sparse polynomial adapter variables must be unique.",
    sparseAdapterVariablesIncludeEvery: "Sparse polynomial adapter variables must include every polynomial variable.",
    sparseExpressionConversionNameCount: "Sparse polynomial expression conversion requires one name per ring variable.",
    converterMissingArgumentScope: "Missing argument scope for function {{function}}",
    converterMissingPrefixOperand: "Missing operand for prefix operator {{operator}}",
    converterMissingOperandMetadata: "Missing operand or metadata for operator {{operator}}",
    converterMissingLeftOperand: "Missing left operand for operator {{operator}}",
    converterTexConversionFailed: "Unable to convert source expression to TeX",
    unknownOperator: "Unknown operator {{operator}}",
    polynomialMaximumIterations: "Maximum iterations reached",
    polyRequires: "{{operation}} requires {{requirement}}.",
    polyVariablesUnique: "{{operation}} variables must be unique.",
    polyVariablesRingOrder: "{{operation}} variables must follow polynomial-ring order.",
    polyVariablesIncludeActive: "{{operation}} variables must include every active variable.",
    polyInputsSelectedOnly: "{{operation}} inputs may use only the selected variables.",
    polyCannotFactorZeroPolynomial: "{{operation}} cannot factor the zero polynomial.",
    polyCannotFactorZero: "{{operation}} cannot factor zero.",
    polyModulusGreaterThanOne: "ModularSparsePolynomial modulus must be greater than one.",
    polyVariableCountMatch: "ModularSparsePolynomial variableCount must match the supplied sparse polynomial.",
    polyInterpolationCannotDepend: "ModularSparsePolynomial interpolation samples cannot depend on the interpolated variable.",
    polyInterpolationInvertibleDifferences: "ModularSparsePolynomial interpolation points must have invertible pairwise differences.",
    polyCoefficientExponentNonnegative: "ModularSparsePolynomial coefficient exponent must be non-negative.",
    polyCannotDivideZero: "ModularSparsePolynomial cannot divide by the zero polynomial.",
    polyDivisorLeadingCoefficientInvertible: "ModularSparsePolynomial divisor leading coefficient is not invertible.",
    polyLeadingCoefficientInvertible: "ModularSparsePolynomial leading coefficient is not invertible.",
    polyExponentNonnegative: "ModularSparsePolynomial exponent must be non-negative.",
    polyDegreeMatrixLimit: "Modular factorization degree exceeds the supported matrix dimension.",
    polyExtendedGcdTwoZeroUndefined: "Modular extended GCD is undefined for two zero polynomials.",
    polyHenselPrimeExceedsTailDegree: "Multivariate Hensel lifting prime must exceed every tail-variable degree.",
    polyHenselTargetNotSmaller: "Hensel lifting target modulus cannot be smaller than the base prime.",
    polyHenselTargetRepeatedSquaring: "Hensel lifting target modulus must be obtained by repeated squaring of the base prime.",
    polyHenselFactorsReconstructModuloBase: "Hensel lifting factors do not reconstruct the integer polynomial modulo the base prime.",
    polyZassenhausFactorsReconstructModuloLift: "Zassenhaus recombination factors do not reconstruct the input modulo the lift modulus.",
    polyWangCannotSpecializeZero: "Wang evaluation search cannot specialize zero.",
    polyModularContentExactDivision: "Modular polynomial content must divide the polynomial exactly.",
    factorizationNoFactors: "Nonconstant factorization returned no factors.",
    polyCrtModuliInvertible: "Polynomial CRT moduli must be invertible.",
    polyIntegerFactorPrimeZero: "Integer factorization cannot choose a prime for zero.",
    polySquareFreeNonExactDivision: "Square-free factorization encountered a non-exact polynomial division.",
    polyZassenhausLostRemaining: "Zassenhaus recombination lost the remaining polynomial.",
    polyZassenhausReconstructionFailed: "Zassenhaus recombination factors do not reconstruct the integer polynomial.",
    polyIntegerFactorLostLeading: "Integer factorization lost the leading term.",
    parserMalformedSymbolicAccessor: "Malformed internal symbolic accessor.",
    parserMalformedSymbolicAccessorIndex: "Malformed internal symbolic accessor index.",
    unknownOperatorAction: "Unknown operator action {{action}}",
    parserBracketTokenMissingCharacter: "Bracket token is missing its character.",
    polyWangLeadingEvaluationInteger: "Wang leading-coefficient evaluation did not produce an integer.",
    polyWangSpecializedLeadingInteger: "Wang specialized factor did not have an integer leading coefficient.",
    polyMultivariateContentDivisionFailed: "Multivariate factorization could not divide a square-free block by its main-variable content.",
    polyMultivariateContentVariableMissing: "Multivariate main-variable content did not retain an active coefficient variable.",
    polyMultivariateReconstructionFailed: "Complete multivariate factorization does not reconstruct the input polynomial.",
    polyBerlekampNoninvertiblePivot: "Berlekamp elimination encountered a noninvertible pivot.",
    polyHenselZeroFactorGroup: "Hensel lifting cannot lift a zero factor group.",
    polyHenselCannotLiftZero: "Hensel lifting cannot lift the zero polynomial.",
    polyHenselModulusExceeded: "Hensel lifting exceeded the requested prime-power modulus.",
    polyHenselLeadingLost: "Hensel lifting lost the target leading coefficient.",
    polyHenselReconstructionFailed: "Hensel-lifted factors do not reconstruct the target polynomial.",
    polyBerlekampNonExactSplit: "Berlekamp splitting produced a non-exact factor division.",
    polyBerlekampIncompleteSplit: "Berlekamp factorization did not fully split the polynomial.",
    polyBerlekampReconstructionFailed: "Berlekamp factors do not reconstruct the input polynomial.",
    expressionKeyValueUnsupported: "The function 'keyValue' is not implemented for type {{type}}.",
    solveLinearNonlinearDegree: "Nonlinear term detected in equation {{equation}}: degree {{degree}} term found. solveLinearSystem only handles linear systems.",
    solveLinearNonPolynomialVariable: "Nonlinear term detected in equation {{equation}}: variable \"{{variable}}\" appears in a non-polynomial position. solveLinearSystem only handles linear systems.",
    polyReqPrimeModulus: "a prime modulus",
    polyReqUnivariatePolynomial: "a univariate polynomial",
    polyReqSquareFreeInput: "square-free input",
    polyReqSamePolynomialRing: "the same polynomial ring",
    polyReqAtLeastOneVariable: "at least one variable",
    polyReqAtLeastTwoSelectedVariables: "at least two selected variables",
    polyReqOneEvaluationPerTailVariable: "one evaluation value per tail variable",
    polyReqMatchingFactorLeadingLists: "matching factor and leading-coefficient lists",
    polyReqPairwiseCoprimeModularFactors: "pairwise coprime modular factors",
    polyReqPrimeBaseModulus: "a prime base modulus",
    polyReqAtLeastOneModularFactor: "at least one modular factor",
    polyReqUnivariateIntegerPolynomial: "a univariate integer polynomial",
    polyReqFactorsSameRingAndPrime: "factors from the same integer polynomial ring and base prime",
    polyReqUnivariateFactors: "univariate factors",
    polyReqMonicModularFactors: "monic modular factors",
    polyReqPrimitiveIntegerPolynomial: "a primitive integer polynomial",
    polyReqPositiveLeadingCoefficient: "a positive leading coefficient",
    polyReqLiftedModularFactors: "lifted modular factors",
    polyReqLiftedFactorsSameRing: "lifted factors from one polynomial ring",
    polyReqUnivariateLiftedFactors: "univariate lifted factors",
    polyReqMonicLiftedFactors: "monic lifted factors",
    polyReqLargerHenselModulus: "a larger Hensel lifting modulus",
    polyReqPrimitivePolynomial: "a primitive polynomial",
    polyReqPositiveLexLeadingSign: "positive lexicographic leading sign",
    polyReqSquareFreeMainVariable: "square-free input in the main variable",
    polyReqExactlyTwoSelectedVariables: "exactly two selected variables",
    polyReqExactlyTwoVariables: "exactly two variables",
    polyReqOneValuePerNonMainVariable: "one value for every non-main selected variable",
    polyReqOneValuePerTailVariable: "one value for every tail variable",
    polyReqPositiveDegreeMainVariable: "positive degree in the main variable",
    polyReqAllOtherVariablesDormant: "all other variables to be dormant",
    polyReqPositiveModuli: "positive moduli",
    polyReqCoprimeModuli: "coprime moduli",
    polyReqAtLeastOneSample: "at least one sample",
    polyReqInvertibleLeadingCoefficient: "an invertible leading coefficient",
    polyReqIntegerLeadingCoefficientInvertible: "the integer leading coefficient to be invertible modulo the base prime",
    infinityMinusInfinityUndefined: 'Infinity-Infinity is undefined!',
    // Assumptions
    inconsistentAssumptions: 'Assumptions are inconsistent (empty intersection)!',
    // Parser
    unsupportedType: 'The provided type ({{type}}) is not supported for this function!',
    unsupportedOperation: 'This operation is currently not supported!',
    buildFunctionUnsupportedFunction: 'buildFunction cannot compile the function "{{function}}" to JavaScript number arithmetic!',
    buildFunctionComplexUnsupported: 'buildFunction cannot compile expressions containing the imaginary unit to JavaScript number arithmetic!',
    nonMatchingDimensions: "This operation cannot be completed because the dimensions don't match!",
    malformedExpression: 'Unable to parse malformed expression!',
    unsupportedCharacter: 'Unsupported character "{{character}}" at position {{position}}.',
    missingCommentDelimiter: 'Missing closing comment delimiter "{{delimiter}}".',
    divisionByZero: 'Division by zero is not defined!',
    zeroToZeroPower: '0^0 is undefined!',
    infinityToPowerZero: 'Infinity^0 is not defined!',
    infinityTimesZero: '0*Infinity is not defined!',
    infinityToInfinity: 'Infinity^Infinity is not defined!',
    valueToInfinityUndefined: '{{value}}^Infinity is not defined!',
    atan2Undefined: 'atan2 is undefined for 0, 0!',
    tanUndefined: 'tan is undefined for multiples of pi/2!',
    loopIterationLimitExceeded: 'Loop exceeded the maximum of {{max}} iterations.',
    breakOutsideLoop: 'break can only be used inside a loop.',
    continueOutsideLoop: 'continue can only be used inside a loop.',
    letRequiresBindingsAndBody: 'let requires one or more name/value pairs followed by a body expression.',
    secUndefined: 'sec is undefined for odd multiples of pi/2!',
    cscUndefined: 'csc is undefined for multiples of pi!',
    cotUndefined: 'cot is undefined for multiples of pi!',
    cothUndefined: 'coth is undefined for 0!',
    cschUndefined: 'csch is undefined for 0!',
    atanhUndefined: 'atanh is undefined for 0!',
    cannotCreateArrayFromNaN: 'Cannot create array from NaN!',
    expressionExpected: 'Expression expected!. Received {{type}}!',
    // TeX Converter.
    unrecognizedMode: '"{{mode}}" is is not a valid mode!',
    // Polynomials
    notAPolynomial: 'The expression is not a valid polynomial!',
    multidegreeMismatch: 'The number of variables must match the multidegree of the polynomial!',
    // Matrix
    cannotCreateMatrix: 'Unable to create Matrix. Row dimensions do not match!',
    cannotMultiplyMatrix: 'Cannot multiply matrix!',
    squareMatrixRequired: 'The matrix must be square!',
    rowsMustMatch: 'The rows of the matrices must match!',
    columnsMustMatch: 'The columns of the matrices must match!',
    singularMatrix: 'The provided matrix is singular!',
    matrixExpected: 'Matrix expected! Received {{type}}.',
    // Vector
    mismatchedDimensions: 'The dimensions must match for the function "{{function}}"!',
    incorrectCrossDimension: 'The cross product is only defined for vectors of dimension 3!',
    vectorExpected: 'Expected Vector! Received {{type}}.',
    // Polynomial
    univariatePolynomialOnly: 'This function is only supported for univariate polynomials!',
    unknownVariable: 'Unknown variable! "{{variable}}"',
    // Solve
    convergenceFailed: 'Failed to converge! {{iters}} iterations were tried.',
    endPointIsRoot: 'End point is root.',
    zeroDerivative: 'The derivative is zero at x = {{x}}. Cannot continue.',
    tooManyUnknowns: 'Too many unknowns!',
    solveSystemInfiniteSolutions: 'This system does not have a finite isolated solution set!',
    solveSystemPolynomialOnly: 'solveSystem currently supports only linear or polynomial systems!',
    solveSystemNonRationalUnsupported: 'This nonlinear polynomial system requires non-rational or non-symbolic solving, which is not yet supported!',
    // Set function
    setJSFunctionExpectsFunction: 'The function parameter when setting a JS function must be of type function!',
    functionRequiresMinAndMaxArgs: 'You must provide a min and max arg when setting the function!',
};
exports.ErrorMessages = {
    eng: exports.EnglishErrorMessages,
};
/** Registers one optional localized message catalog. */
function registerLanguage(language, catalog) {
    exports.ErrorMessages[language] = catalog;
}
/** Formats an internal localized error message with named template values. */
function message(e, values) {
    const catalog = exports.ErrorMessages[Settings_1.Settings.LANGUAGE] ?? exports.ErrorMessages.eng;
    return (0, string_1.format)(catalog[e], values || {});
}

};
modules["core/Settings"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SPECIAL = exports.RESTRICTED = exports.Settings = void 0;
exports.scopedBlock = scopedBlock;
const constants_1 = require("./classes/parser/constants");
exports.Settings = {
    // Forces decimals to fractions.
    USE_FRACTIONS: true,
    // Allow bigInt to be used when numerator or denominator get huge
    ALLOW_BIGINT: true,
    // If this is true then functions and numbers will be evaluated
    EVALUATE: false,
    // If true variable substitution will be performed
    SUBSTITUTE: true,
    // The language being used
    LANGUAGE: 'eng',
    // Allows implicit multiplications
    ALLOW_IMPLICIT_MULTIPLICATION: true,
    // If true the terms will be sorted. The default is false because of the added overhead.
    SORT_TERMS: false,
    // Avoids certain simplifications if set to false for performance boots
    DEFER_SIMPLIFICATION: false,
    // If true powers of 1/2 will be rewritten back to sqrt
    REWRITE_SQRT: true,
    // The max allowable integer when factoring numerators and denominator
    MAX_FRAC_INT: BigInt(1e60),
    // If set to false then rational multipliers will not be allowed in substitutions
    ALLOW_RAT_SUBS: false,
    // If set to true, vectors, matrices, and collections will always return a copy
    IMMUTABLE_VALUE_SETS: true,
    // If true each letter will be treated a a variable
    USE_SINGLE_LETTER_VARIABLES: false,
    // The maximum iteration allowed in sum and product
    MAX_PRODUCT_AND_SUMMATION_ITERATION: 1e4,
    // The maximum number of completed iterations allowed for parser control-flow loops
    MAX_LOOP_ITERATIONS: 1e4,
    // If true then rules will be initialized on start
    INITIALIZE_RULES: true,
    // The base index for element access on vectors and matrices. 0 = zero-based, 1 = one-based.
    INDEX_BASE: 0,
};
/**
 * Allows for the parser to run with a desired setting for a single call
 *
 * @param setting The setting to modified for the call
 * @param value The value of the setting under which the function is called
 * @param callback The function to be called
 * @returns The result of the call
 */
function scopedBlock(setting, value, callback) {
    let retval;
    // Store the value being currently used
    const currentSettingValue = exports.Settings[setting];
    // Update the setting to the desired value
    exports.Settings[setting] = value;
    try {
        retval = callback();
    }
    finally {
        // Restore the value even if the callback throws
        exports.Settings[setting] = currentSettingValue;
    }
    return retval;
}
// The list of restricted variables
exports.RESTRICTED = [constants_1.E, constants_1.PI_NAME, constants_1.DEFAULT_IMAGINARY, constants_1.INDEX_VARIABLE, constants_1.ALL_SYMBOL];
// The list of special values that get ignore at substitution
exports.SPECIAL = [constants_1.E, constants_1.PI_NAME];

};
modules["core/classes/parser/constants"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FIB = exports.DOUBLE_FACTORIAL = exports.FACTORIAL = exports.SYMBOLIC_ACCESSOR = exports.PARSER_CONSTANTS = exports.INFINITY = exports.PI = exports.ALIASES = exports.ALL_SYMBOL = exports.INDEX_VARIABLE = exports.DEFAULT_IMAGINARY = exports.PI_SYMBOL = exports.PI_NAME = exports.E = exports.RANK = exports.EXPRESSION_TYPES = exports.dataTypes = exports.ASSUMPTION = exports.DICTIONARY = exports.SOLUTIONS_SET = exports.SET = exports.GFPOLY = exports.TERM = exports.POLYNOMIAL = exports.EQUATION = exports.MATRIX = exports.VECTOR = exports.COLLECTION = exports.EXPRESSION = exports.SCOPE = exports.RATIONAL = exports.KEY_VALUE_PAIR = exports.INDEXED_REFERENCE = exports.TOKEN = exports.WRAP = exports.CLOSE_PAREN = exports.OPEN_PAREN = exports.RETURN = exports.LINEFEED = exports.TAB = exports.NEWLINE = exports.BLANK = exports.SPACE = exports.ASSIGN = exports.COMMA = exports.DOT = exports.DIVIDE = exports.MULTIPLY = exports.PLUS = exports.MINUS = void 0;
exports.SI = exports.SHI = exports.TRUNC = exports.SIZE = exports.ROUND = exports.REALPART = exports.PRODUCT_FUNCTION = exports.NTHROOT = exports.MIN = exports.MAX = exports.LN = exports.LIMIT = exports.LI = exports.LAPLACE = exports.ISPRIME = exports.INTEGRATE = exports.IMATRIX = exports.IMAGPART = exports.ILAPLACE = exports.GCD = exports.FRESNEL_S = exports.FRESNEL_C = exports.FLOOR = exports.PFACTORD = exports.PFACTOR = exports.FACTOR = exports.ERF = exports.EI = exports.DOT_PRODUCT = exports.DIFF = exports.DEG = exports.DEFINT = exports.CROSS = exports.COUNT = exports.CONTENT = exports.CONTAINS = exports.CONJUGATE = exports.CI = exports.CHI = exports.CEIL = exports.CBRT = exports.CSGN = exports.SGN = exports.LOG = exports.HEAVISIDE = exports.DIRAC = exports.GAMMA = exports.MOD = exports.ABS = exports.SQRT = void 0;
exports.INVERSE_HYPERBOLIC_TRIG = exports.HYPERBOLIC_TRIG = exports.INVERSE_TRIG = exports.TRIG = exports.ACOTH = exports.ACSCH = exports.ASECH = exports.ATANH = exports.ASINH = exports.ACOSH = exports.COTH = exports.CSCH = exports.SECH = exports.TANH = exports.COSH = exports.SINH = exports.ATAN2 = exports.ACOT = exports.ACSC = exports.ASEC = exports.ATAN = exports.COT = exports.CSC = exports.SEC = exports.ACOS = exports.ASIN = exports.TAN = exports.SIN = exports.COS = exports.TRIG_FUNCTION_NAMES = exports.SUM_FUNCTION = exports.SINC = void 0;
const Decimal = { acos: x => ({ toString: () => String(Math.acos(x)) }), exp: x => ({ toString: () => String(Math.exp(x)) }) };
/**
 * Some helper constants
 */
exports.MINUS = '-';
exports.PLUS = '+';
exports.MULTIPLY = '*';
exports.DIVIDE = '/';
exports.DOT = '.';
exports.COMMA = ',';
exports.ASSIGN = ':';
exports.SPACE = ' ';
exports.BLANK = '';
exports.NEWLINE = '\n';
exports.TAB = '\t';
exports.LINEFEED = '\f';
exports.RETURN = '\r';
exports.OPEN_PAREN = '(';
exports.CLOSE_PAREN = ')';
exports.WRAP = '';
// Data types
// export const EXPRESSION = 'EX';
exports.TOKEN = 'Token';
exports.INDEXED_REFERENCE = 'INDEXED_REF';
exports.KEY_VALUE_PAIR = 'KEY_VALUE_PAIR';
exports.RATIONAL = 'RAT';
exports.SCOPE = 'SCP';
exports.EXPRESSION = 'EXP';
exports.COLLECTION = 'COL';
exports.VECTOR = 'VEC';
exports.MATRIX = 'MAT';
exports.EQUATION = 'EQN';
exports.POLYNOMIAL = 'POL';
exports.TERM = 'TRM';
exports.GFPOLY = 'GFP';
exports.SET = 'SET';
exports.SOLUTIONS_SET = 'SOL';
exports.DICTIONARY = 'DIC';
exports.ASSUMPTION = 'ASSUMPTION';
// The data type name
exports.dataTypes = {
    [exports.TOKEN]: 'Token',
    [exports.INDEXED_REFERENCE]: 'IndexedReference',
    [exports.KEY_VALUE_PAIR]: 'KeyValuePair',
    [exports.RATIONAL]: 'Rational',
    [exports.SCOPE]: 'Scope',
    [exports.EXPRESSION]: 'Expression',
    [exports.COLLECTION]: 'Collection',
    [exports.VECTOR]: 'Vector',
    [exports.MATRIX]: 'Matrix',
    [exports.EQUATION]: 'Equation',
    [exports.POLYNOMIAL]: 'Polynomial',
    [exports.TERM]: 'Term',
    [exports.GFPOLY]: 'GFPolynomial',
    [exports.SET]: 'ValuesSet',
    [exports.SOLUTIONS_SET]: 'SolutionSet',
    [exports.DICTIONARY]: 'Dictionary',
    [exports.ASSUMPTION]: 'Assumption',
};
/**
 * The base types supported by the parser. All other types are a form of these
 */
exports.EXPRESSION_TYPES = {
    // Numbers,
    NUM: 6,
    // Any variable e.g. x
    VAR: 2,
    // An exponential
    EXP: 1,
    // Any function
    FUN: 3,
    // Any type of the same variable
    GRP: 4,
    // Any expression with terms bound by multiplication or division
    PRD: 7,
    // Any expression bound by division or subtraction
    SUM: 5,
    // Infinity
    INF: 8,
    // Vectors
    VEC: 9,
    // Matrices
    MAT: 10,
};
/**
 * The ranking used when evaluating types
 */
exports.RANK = {
    // Numbers,
    [exports.EXPRESSION_TYPES.NUM]: 1,
    // Any variable e.g. x
    [exports.EXPRESSION_TYPES.VAR]: 2,
    // An exponential
    [exports.EXPRESSION_TYPES.EXP]: 3,
    // Any function
    [exports.EXPRESSION_TYPES.FUN]: 4,
    // Any type of the same variable
    [exports.EXPRESSION_TYPES.GRP]: 5,
    // Any expression with terms bound by multiplication or division
    [exports.EXPRESSION_TYPES.PRD]: 6,
    // Any expression bound by division or subtraction
    [exports.EXPRESSION_TYPES.SUM]: 7,
    // Infinity
    [exports.EXPRESSION_TYPES.INF]: 8,
};
// Parser Constants
exports.E = 'e';
exports.PI_NAME = 'pi';
exports.PI_SYMBOL = 'π';
exports.DEFAULT_IMAGINARY = 'i';
exports.INDEX_VARIABLE = '_n';
exports.ALL_SYMBOL = 'all';
exports.ALIASES = {
    [exports.PI_SYMBOL]: exports.PI_NAME,
    '∞': 'Inf',
};
exports.PI = [exports.PI_NAME, exports.PI_SYMBOL];
exports.INFINITY = ['Inf', 'Infinity', '∞'];
// Parser Constants
exports.PARSER_CONSTANTS = {
    [exports.PI_NAME]: () => {
        return Decimal.acos(-1).toString();
    },
    [exports.PI_SYMBOL]: () => {
        return Decimal.acos(-1).toString();
    },
    [exports.E]: () => {
        return Decimal.exp(1).toString();
    },
};
// Functions
// Internal function name used to carry symbolic bracket access inside an Expression.
exports.SYMBOLIC_ACCESSOR = '__accessor__';
exports.FACTORIAL = 'fact';
exports.DOUBLE_FACTORIAL = 'dfact';
exports.FIB = 'fib';
exports.SQRT = 'sqrt';
exports.ABS = 'abs';
exports.MOD = 'mod';
exports.GAMMA = 'gamma';
exports.DIRAC = 'delta';
exports.HEAVISIDE = 'heaviside';
exports.LOG = 'log';
exports.SGN = 'sign';
exports.CSGN = 'csgn';
exports.CBRT = 'cbrt';
exports.CEIL = 'ceil';
exports.CHI = 'Chi';
exports.CI = 'Ci';
exports.CONJUGATE = 'conjugate';
exports.CONTAINS = 'contains';
exports.CONTENT = 'content';
exports.COUNT = 'count';
exports.CROSS = 'cross';
exports.DEFINT = 'defint';
exports.DEG = 'deg';
exports.DIFF = 'diff';
exports.DOT_PRODUCT = 'dot';
exports.EI = 'Ei';
exports.ERF = 'erf';
exports.FACTOR = 'factor';
exports.PFACTOR = 'pfactor';
exports.PFACTORD = 'pfactord';
exports.FLOOR = 'floor';
exports.FRESNEL_C = 'C';
exports.FRESNEL_S = 'S';
exports.GCD = 'gcd';
exports.ILAPLACE = 'ilaplace';
exports.IMAGPART = 'imagpart';
exports.IMATRIX = 'imatrix';
exports.INTEGRATE = 'integrate';
exports.ISPRIME = 'isprime';
exports.LAPLACE = 'laplace';
exports.LI = 'Li';
exports.LIMIT = 'limit';
exports.LN = 'ln';
exports.MAX = 'max';
exports.MIN = 'min';
exports.NTHROOT = 'nthroot';
exports.PRODUCT_FUNCTION = 'product';
exports.REALPART = 'realpart';
exports.ROUND = 'round';
exports.SIZE = 'size';
exports.TRUNC = 'trunc';
exports.SHI = 'Shi';
exports.SI = 'Si';
exports.SINC = 'sinc';
exports.SUM_FUNCTION = 'sum';
// Trigonometric functions
exports.TRIG_FUNCTION_NAMES = {
    COS: 'cos',
    SIN: 'sin',
    TAN: 'tan',
    SEC: 'sec',
    CSC: 'csc',
    COT: 'cot',
    ACOS: 'acos',
    ASIN: 'asin',
    ATAN: 'atan',
    ASEC: 'asec',
    ACSC: 'acsc',
    ACOT: 'acot',
    ATAN2: 'atan2',
    SINH: 'sinh',
    COSH: 'cosh',
    TANH: 'tanh',
    SECH: 'sech',
    CSCH: 'csch',
    COTH: 'coth',
    ACOSH: 'acosh',
    ASINH: 'asinh',
    ATANH: 'atanh',
    ASECH: 'asech',
    ACSCH: 'acsch',
    ACOTH: 'acoth',
};
exports.COS = exports.TRIG_FUNCTION_NAMES.COS, exports.SIN = exports.TRIG_FUNCTION_NAMES.SIN, exports.TAN = exports.TRIG_FUNCTION_NAMES.TAN, exports.ASIN = exports.TRIG_FUNCTION_NAMES.ASIN, exports.ACOS = exports.TRIG_FUNCTION_NAMES.ACOS, exports.SEC = exports.TRIG_FUNCTION_NAMES.SEC, exports.CSC = exports.TRIG_FUNCTION_NAMES.CSC, exports.COT = exports.TRIG_FUNCTION_NAMES.COT, exports.ATAN = exports.TRIG_FUNCTION_NAMES.ATAN, exports.ASEC = exports.TRIG_FUNCTION_NAMES.ASEC, exports.ACSC = exports.TRIG_FUNCTION_NAMES.ACSC, exports.ACOT = exports.TRIG_FUNCTION_NAMES.ACOT, exports.ATAN2 = exports.TRIG_FUNCTION_NAMES.ATAN2, exports.SINH = exports.TRIG_FUNCTION_NAMES.SINH, exports.COSH = exports.TRIG_FUNCTION_NAMES.COSH, exports.TANH = exports.TRIG_FUNCTION_NAMES.TANH, exports.SECH = exports.TRIG_FUNCTION_NAMES.SECH, exports.CSCH = exports.TRIG_FUNCTION_NAMES.CSCH, exports.COTH = exports.TRIG_FUNCTION_NAMES.COTH, exports.ACOSH = exports.TRIG_FUNCTION_NAMES.ACOSH, exports.ASINH = exports.TRIG_FUNCTION_NAMES.ASINH, exports.ATANH = exports.TRIG_FUNCTION_NAMES.ATANH, exports.ASECH = exports.TRIG_FUNCTION_NAMES.ASECH, exports.ACSCH = exports.TRIG_FUNCTION_NAMES.ACSCH, exports.ACOTH = exports.TRIG_FUNCTION_NAMES.ACOTH;
exports.TRIG = [exports.COS, exports.SIN, exports.TAN, exports.SEC, exports.CSC, exports.COT];
exports.INVERSE_TRIG = [exports.ACOS, exports.ASIN, exports.ATAN, exports.ASEC, exports.ACSC, exports.ACOT];
exports.HYPERBOLIC_TRIG = [exports.COSH, exports.SINH, exports.TANH, exports.SECH, exports.CSCH, exports.COTH];
exports.INVERSE_HYPERBOLIC_TRIG = [exports.ACOSH, exports.ASINH, exports.ATANH, exports.ASECH, exports.ACSCH, exports.ACOTH];

};
modules["algebra/polynomial/SparsePolynomialMultivariateFactor"]=function(require,module,exports){
"use strict";
/**
 * @module sparsePolynomialMultivariateFactor
 *
 * Evaluation configuration for Wang/EEZ multivariate integer factorization.
 *
 * External references consulted:
 * - Wang/EEZ evaluation-point tests and lifting pipeline:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/factortools.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.testWangEvaluationPoint = testWangEvaluationPoint;
exports.testWangConfiguration = testWangConfiguration;
exports.testWangBivariateConfiguration = testWangBivariateConfiguration;
exports.liftWangConfiguration = liftWangConfiguration;
exports.liftWangBivariateConfiguration = liftWangBivariateConfiguration;
exports.factorSquareFreeMultivariateInteger = factorSquareFreeMultivariateInteger;
exports.factorSquareFreeBivariateInteger = factorSquareFreeBivariateInteger;
exports.factorMultivariateInteger = factorMultivariateInteger;
exports.factorBivariateInteger = factorBivariateInteger;
exports.findWangConfiguration = findWangConfiguration;
exports.findWangBivariateConfiguration = findWangBivariateConfiguration;
exports.findWangEvaluationPoint = findWangEvaluationPoint;
const SparsePolynomial_1 = require("../../core/classes/polynomial/SparsePolynomial");
const errors_1 = require("../../core/errors");
const bigint_1 = require("../../core/functions/bigint/bigint");
const primeFactor_1 = require("../../core/functions/bigint/primeFactor");
const ModularSparsePolynomialFactor_1 = require("./ModularSparsePolynomialFactor");
const SparsePolynomialFactor_1 = require("./SparsePolynomialFactor");
const SparsePolynomialGcd_1 = require("./SparsePolynomialGcd");
const INITIAL_WANG_LIFT_PRIME = 1000003n;
function assertWangVariables(polynomial, variables) {
    if (variables.length < 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang evaluation', requirement: (0, errors_1.message)('polyReqAtLeastTwoSelectedVariables') }));
    }
    const selected = new Set();
    let previous = -1;
    for (const variableIndex of variables) {
        polynomial.degree(variableIndex);
        if (selected.has(variableIndex)) {
            throw new RangeError((0, errors_1.message)('polyVariablesUnique', { operation: 'Wang evaluation' }));
        }
        if (variableIndex <= previous) {
            throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Wang evaluation' }));
        }
        selected.add(variableIndex);
        previous = variableIndex;
    }
    for (const activeVariable of polynomial.variables()) {
        if (!selected.has(activeVariable)) {
            throw new RangeError((0, errors_1.message)('polyVariablesIncludeActive', { operation: 'Wang evaluation' }));
        }
    }
}
function evaluateTail(polynomial, variables, values) {
    return polynomial.evaluateVariables(variables.slice(1), values);
}
function normalizeSpecialization(polynomial) {
    let content = polynomial.content();
    let primitive = polynomial.primitivePart();
    const leading = primitive.leadingTerm('lex');
    if (leading !== null && leading.coefficient < 0n) {
        content = -content;
        primitive = primitive.negate();
    }
    return { content, primitive };
}
function evaluateTailInteger(polynomial, variables, values) {
    const evaluated = evaluateTail(polynomial, variables, values);
    if (!evaluated.isConstant()) {
        throw new Error((0, errors_1.message)('polyWangLeadingEvaluationInteger'));
    }
    return evaluated.constantTerm();
}
function wangNonDivisors(values, specializedContent, leadingContent) {
    const previous = [specializedContent * leadingContent];
    for (const value of values) {
        let remaining = value < 0n ? -value : value;
        for (let index = previous.length - 1; index >= 0; index--) {
            let divisor = previous[index];
            if (divisor < 0n) {
                divisor = -divisor;
            }
            while (divisor !== 1n) {
                divisor = (0, bigint_1.GCD)(divisor, remaining);
                remaining /= divisor;
            }
        }
        if (remaining === 1n) {
            return false;
        }
        previous.push(value < 0n ? -value : value);
    }
    return true;
}
function productPolynomials(polynomials, variableCount) {
    let result = SparsePolynomial_1.SparsePolynomial.constant(variableCount, 1n);
    for (const polynomial of polynomials) {
        result = result.multiply(polynomial);
    }
    return result;
}
function multivariateMignotteBound(polynomial, variables) {
    let degreeSum = 0n;
    for (const variableIndex of variables) {
        degreeSum += polynomial.degree(variableIndex) ?? 0n;
    }
    const leading = polynomial.leadingTerm('lex');
    if (leading === null) {
        return 0n;
    }
    const leadingAbsolute = leading.coefficient < 0n ? -leading.coefficient : leading.coefficient;
    return ((0, bigint_1.ceilIntegerSquareRoot)(degreeSum + 1n) *
        (1n << degreeSum) *
        polynomial.maxAbsoluteCoefficient() *
        leadingAbsolute);
}
function testWangSpecializationPrepared(polynomial, variables, values, mainVariable, originalDegree) {
    const evaluated = evaluateTail(polynomial, variables, values);
    if (evaluated.degree(mainVariable) !== originalDegree) {
        return null;
    }
    const normalized = normalizeSpecialization(evaluated);
    if (normalized.primitive.isConstant()) {
        return null;
    }
    const derivative = normalized.primitive.derivative(mainVariable);
    const gcd = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(normalized.primitive, derivative, [mainVariable]);
    if (!gcd.isConstant()) {
        return null;
    }
    return {
        values: [...values],
        content: normalized.content,
        polynomial: normalized.primitive,
    };
}
function factorWangSpecialization(specialization, mainVariable) {
    return {
        ...specialization,
        factors: (0, SparsePolynomialFactor_1.factorSquareFreeUnivariateInteger)(specialization.polynomial, mainVariable),
    };
}
/**
 * Tests one Wang/EEZ specialization.
 *
 * The first selected variable remains symbolic and every later variable is replaced by
 * the corresponding integer value. A usable specialization preserves the degree in the
 * main variable and has square-free primitive part.
 */
function testWangEvaluationPoint(polynomial, variables, values) {
    assertWangVariables(polynomial, variables);
    if (values.length !== variables.length - 1) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang evaluation', requirement: (0, errors_1.message)('polyReqOneValuePerNonMainVariable') }));
    }
    if (polynomial.isZero()) {
        return null;
    }
    const mainVariable = variables[0];
    const originalDegree = polynomial.degree(mainVariable);
    if (originalDegree === null || originalDegree === 0n) {
        return null;
    }
    const specialization = testWangSpecializationPrepared(polynomial, variables, values, mainVariable, originalDegree);
    return specialization === null
        ? null
        : factorWangSpecialization(specialization, mainVariable);
}
function visitEvaluationShell(valueCount, radius, visitor) {
    const values = new Array(valueCount).fill(0n);
    function visit(index, touchesBoundary) {
        if (index === valueCount) {
            return touchesBoundary && visitor(values);
        }
        for (let value = -radius; value <= radius; value++) {
            casBudgetCheck();
            values[index] = value;
            if (visit(index + 1, touchesBoundary || value === -radius || value === radius)) {
                return true;
            }
        }
        return false;
    }
    return visit(0, radius === 0n);
}
function factorWangLeadingCoefficient(polynomial, variables) {
    const mainVariable = variables[0];
    const tailVariables = variables.slice(1);
    const leadingCoefficient = polynomial.leadingCoefficientIn(mainVariable);
    if (leadingCoefficient === null) {
        return null;
    }
    const activeTailVariables = tailVariables.filter(variableIndex => (leadingCoefficient.degree(variableIndex) ?? 0n) > 0n);
    if (activeTailVariables.length <= 1) {
        return (0, SparsePolynomialFactor_1.factorUnivariateInteger)(leadingCoefficient, activeTailVariables[0] ?? tailVariables[0]);
    }
    return factorMultivariateInteger(leadingCoefficient, activeTailVariables);
}
function testWangConfigurationWithLeadingFactorization(polynomial, variables, values, leadingFactorization, mainVariable, originalDegree) {
    const specialization = testWangSpecializationPrepared(polynomial, variables, values, mainVariable, originalDegree);
    if (specialization === null) {
        return null;
    }
    const leadingValues = leadingFactorization.factors.map(entry => evaluateTailInteger(entry.factor, variables, values));
    if (!wangNonDivisors(leadingValues, specialization.content, leadingFactorization.content)) {
        return null;
    }
    const evaluation = factorWangSpecialization(specialization, mainVariable);
    const assigned = [];
    const usedLeadingFactors = new Array(leadingFactorization.factors.length).fill(false);
    let remainingContent = evaluation.content;
    const specializedFactors = evaluation.factors.map(factor => factor);
    for (const specializedFactor of specializedFactors) {
        const factorLeading = specializedFactor.leadingCoefficientIn(mainVariable);
        if (factorLeading === null || !factorLeading.isConstant()) {
            throw new Error((0, errors_1.message)('polyWangSpecializedLeadingInteger'));
        }
        let dividend = factorLeading.constantTerm() * remainingContent;
        let assignedLeading = SparsePolynomial_1.SparsePolynomial.constant(polynomial.variableCount, 1n);
        for (let index = leadingFactorization.factors.length - 1; index >= 0; index--) {
            const evaluatedFactor = leadingValues[index];
            if (evaluatedFactor === 0n) {
                return null;
            }
            let exponent = 0n;
            while (dividend % evaluatedFactor === 0n) {
                dividend /= evaluatedFactor;
                exponent++;
            }
            if (exponent > 0n) {
                assignedLeading = assignedLeading.multiply(leadingFactorization.factors[index].factor.pow(exponent));
                usedLeadingFactors[index] = true;
            }
        }
        assigned.push(assignedLeading);
    }
    if (usedLeadingFactors.some(used => !used)) {
        return null;
    }
    const adjustedFactors = [];
    const adjustedLeading = [];
    for (let index = 0; index < assigned.length; index++) {
        let leading = assigned[index];
        let specializedFactor = specializedFactors[index];
        const evaluatedLeading = evaluateTailInteger(leading, variables, values);
        const specializedLeading = specializedFactor
            .leadingCoefficientIn(mainVariable)
            ?.constantTerm();
        if (specializedLeading === undefined || evaluatedLeading === 0n) {
            return null;
        }
        if (remainingContent === 1n) {
            if (specializedLeading % evaluatedLeading !== 0n) {
                return null;
            }
            leading = leading.scale(specializedLeading / evaluatedLeading);
        }
        else {
            const divisor = (0, bigint_1.GCD)(specializedLeading, evaluatedLeading);
            const factorScale = evaluatedLeading / divisor;
            const leadingScale = specializedLeading / divisor;
            if (factorScale === 0n || remainingContent % factorScale !== 0n) {
                return null;
            }
            specializedFactor = specializedFactor.scale(factorScale);
            remainingContent /= factorScale;
            leading = leading.scale(leadingScale);
        }
        adjustedFactors.push(specializedFactor);
        adjustedLeading.push(leading);
    }
    let targetPolynomial = polynomial;
    if (remainingContent !== 1n) {
        adjustedFactors.forEach((factor, index) => {
            adjustedFactors[index] = factor.scale(remainingContent);
            adjustedLeading[index] = adjustedLeading[index].scale(remainingContent);
        });
        targetPolynomial = targetPolynomial.scale(remainingContent ** BigInt(adjustedFactors.length - 1));
    }
    const targetLeading = targetPolynomial.leadingCoefficientIn(mainVariable);
    if (targetLeading === null ||
        !productPolynomials(adjustedLeading, polynomial.variableCount).equals(targetLeading)) {
        return null;
    }
    const specializedTarget = evaluateTail(targetPolynomial, variables, values);
    if (!productPolynomials(adjustedFactors, polynomial.variableCount).equals(specializedTarget)) {
        return null;
    }
    for (let index = 0; index < adjustedFactors.length; index++) {
        const factorLeading = adjustedFactors[index].leadingCoefficientIn(mainVariable);
        if (factorLeading === null ||
            !evaluateTail(adjustedLeading[index], variables, values).equals(factorLeading)) {
            return null;
        }
    }
    return {
        evaluation,
        polynomial: targetPolynomial,
        factors: adjustedFactors,
        leadingCoefficients: adjustedLeading,
    };
}
/**
 * Builds a Wang configuration with reconstructed factor leading coefficients.
 *
 * The main-variable leading coefficient is factored recursively over the tail
 * variables. A single active tail variable uses the univariate factorizer; larger
 * coefficient rings recurse through multivariate factorization.
 */
function testWangConfiguration(polynomial, variables, values) {
    assertWangVariables(polynomial, variables);
    if (values.length !== variables.length - 1) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqOneValuePerTailVariable') }));
    }
    if (polynomial.content() !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqPrimitivePolynomial') }));
    }
    const leadingTerm = polynomial.leadingTerm('lex');
    if (leadingTerm === null || leadingTerm.coefficient < 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqPositiveLexLeadingSign') }));
    }
    const mainVariable = variables[0];
    const originalDegree = polynomial.degree(mainVariable);
    if (originalDegree === null || originalDegree === 0n) {
        return null;
    }
    const leadingFactorization = factorWangLeadingCoefficient(polynomial, variables);
    if (leadingFactorization === null) {
        return null;
    }
    return testWangConfigurationWithLeadingFactorization(polynomial, variables, values, leadingFactorization, mainVariable, originalDegree);
}
/** Builds a bivariate Wang configuration through the general reconstruction path. */
function testWangBivariateConfiguration(polynomial, variables, value) {
    if (variables.length !== 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Bivariate Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqExactlyTwoSelectedVariables') }));
    }
    return testWangConfiguration(polynomial, variables, [value]);
}
/** Lifts a Wang configuration using the reusable multivariate Hensel implementation. */
function liftWangConfiguration(configuration, variables, prime) {
    if (variables.length < 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang lifting', requirement: (0, errors_1.message)('polyReqAtLeastTwoSelectedVariables') }));
    }
    return (0, ModularSparsePolynomialFactor_1.henselLiftMultivariateFactors)(configuration.polynomial, configuration.factors, configuration.leadingCoefficients, variables, configuration.evaluation.values, prime);
}
/** Lifts one bivariate Wang configuration modulo a prime. */
function liftWangBivariateConfiguration(configuration, variables, prime) {
    if (variables.length !== 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Bivariate Wang lifting', requirement: (0, errors_1.message)('polyReqExactlyTwoVariables') }));
    }
    return liftWangConfiguration(configuration, variables, prime);
}
/**
 * Factors a primitive square-free multivariate integer polynomial.
 *
 * A Wang configuration supplies the specialized factors and reconstructed leading
 * coefficients. Modular lifts are retried with increasing primes until centered integer
 * factors reconstruct the original polynomial exactly.
 */
function factorSquareFreeMultivariateInteger(polynomial, variables) {
    assertWangVariables(polynomial, variables);
    if (polynomial.isZero()) {
        throw new RangeError((0, errors_1.message)('polyCannotFactorZero', { operation: 'Wang factorization' }));
    }
    if (polynomial.content() !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang factorization', requirement: (0, errors_1.message)('polyReqPrimitivePolynomial') }));
    }
    const leading = polynomial.leadingTerm('lex');
    if (leading === null || leading.coefficient < 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang factorization', requirement: (0, errors_1.message)('polyReqPositiveLexLeadingSign') }));
    }
    const mainVariable = variables[0];
    const derivative = polynomial.derivative(mainVariable);
    if (!(0, SparsePolynomialGcd_1.sparsePolynomialGcd)(polynomial, derivative, variables).isConstant()) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang factorization', requirement: (0, errors_1.message)('polyReqSquareFreeMainVariable') }));
    }
    const { originalDegree, leadingFactorization, } = prepareWangConfigurationSearch(polynomial, variables);
    let radius = 0n;
    let minimumFactorCount = null;
    let configurations = [];
    while (true) {
        casBudgetCheck();
        let irreducible = false;
        visitEvaluationShell(variables.length - 1, radius, values => {
            const candidate = testWangConfigurationWithLeadingFactorization(polynomial, variables, values, leadingFactorization, mainVariable, originalDegree);
            if (candidate === null) {
                return false;
            }
            if (candidate.factors.length === 1) {
                irreducible = true;
                return true;
            }
            if (minimumFactorCount === null ||
                candidate.factors.length < minimumFactorCount) {
                minimumFactorCount = candidate.factors.length;
                configurations = [];
            }
            if (candidate.factors.length !== minimumFactorCount) {
                return false;
            }
            let maxTailDegree = 0n;
            for (const variableIndex of variables.slice(1)) {
                const degree = candidate.polynomial.degree(variableIndex) ?? 0n;
                if (degree > maxTailDegree) {
                    maxTailDegree = degree;
                }
            }
            const coefficientBound = multivariateMignotteBound(candidate.polynomial, variables);
            let primeStart = 2n * coefficientBound + 1n;
            if (primeStart < INITIAL_WANG_LIFT_PRIME) {
                primeStart = INITIAL_WANG_LIFT_PRIME;
            }
            if (primeStart <= maxTailDegree) {
                primeStart = maxTailDegree + 1n;
            }
            configurations.push({
                configuration: candidate,
                prime: (0, primeFactor_1.nextPrimeBig)(primeStart),
            });
            return false;
        });
        if (irreducible) {
            return [polynomial];
        }
        const retry = [];
        for (const state of configurations) {
            const lifted = liftWangConfiguration(state.configuration, variables, state.prime);
            if (lifted === null) {
                retry.push({
                    configuration: state.configuration,
                    prime: (0, primeFactor_1.nextPrimeBig)(state.prime + 1n),
                });
                continue;
            }
            const reconstruction = productPolynomials(lifted, polynomial.variableCount);
            if (reconstruction.equals(polynomial)) {
                return lifted;
            }
        }
        configurations = retry;
        radius++;
    }
}
/** Factors a primitive square-free bivariate polynomial over the integers. */
function factorSquareFreeBivariateInteger(polynomial, variables) {
    if (variables.length !== 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Bivariate Wang factorization', requirement: (0, errors_1.message)('polyReqExactlyTwoSelectedVariables') }));
    }
    return factorSquareFreeMultivariateInteger(polynomial, variables);
}
/**
 * Completely factors a multivariate integer polynomial.
 *
 * Numeric content is returned separately. Main-variable content is factored in the
 * remaining variables before the primitive portion enters Wang lifting.
 */
function factorMultivariateInteger(polynomial, variables) {
    assertWangVariables(polynomial, variables);
    const squareFree = (0, SparsePolynomialFactor_1.squareFreeFactorizationMultivariate)(polynomial, variables);
    if (squareFree.content === 0n || squareFree.factors.length === 0) {
        return squareFree;
    }
    const mainVariable = variables[0];
    const coefficientVariables = variables.slice(1);
    const factors = [];
    for (const entry of squareFree.factors) {
        const variableContent = (0, SparsePolynomialFactor_1.contentInVariable)(entry.factor, mainVariable, coefficientVariables);
        const primitive = entry.factor.divideExact(variableContent);
        if (primitive === null) {
            throw new Error((0, errors_1.message)('polyMultivariateContentDivisionFailed'));
        }
        if (!variableContent.isConstant()) {
            const activeContentVariables = coefficientVariables.filter(variableIndex => (variableContent.degree(variableIndex) ?? 0n) > 0n);
            let contentFactorization;
            if (activeContentVariables.length === 1) {
                contentFactorization = (0, SparsePolynomialFactor_1.factorUnivariateInteger)(variableContent, activeContentVariables[0]);
            }
            else if (activeContentVariables.length > 1) {
                contentFactorization = factorMultivariateInteger(variableContent, activeContentVariables);
            }
            else {
                throw new Error((0, errors_1.message)('polyMultivariateContentVariableMissing'));
            }
            for (const contentEntry of contentFactorization.factors) {
                factors.push({
                    factor: contentEntry.factor,
                    multiplicity: entry.multiplicity * contentEntry.multiplicity,
                });
            }
        }
        if (!primitive.isConstant()) {
            for (const irreducible of factorSquareFreeMultivariateInteger(primitive, variables)) {
                factors.push({
                    factor: irreducible,
                    multiplicity: entry.multiplicity,
                });
            }
        }
    }
    let reconstruction = SparsePolynomial_1.SparsePolynomial.constant(polynomial.variableCount, squareFree.content);
    for (const entry of factors) {
        reconstruction = reconstruction.multiply(entry.factor.pow(entry.multiplicity));
    }
    if (!reconstruction.equals(polynomial)) {
        throw new Error((0, errors_1.message)('polyMultivariateReconstructionFailed'));
    }
    return {
        content: squareFree.content,
        factors,
    };
}
/** Completely factors a bivariate integer polynomial. */
function factorBivariateInteger(polynomial, variables) {
    if (variables.length !== 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Complete bivariate factorization', requirement: (0, errors_1.message)('polyReqExactlyTwoSelectedVariables') }));
    }
    return factorMultivariateInteger(polynomial, variables);
}
function prepareWangConfigurationSearch(polynomial, variables) {
    assertWangVariables(polynomial, variables);
    if (polynomial.content() !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqPrimitivePolynomial') }));
    }
    const leadingTerm = polynomial.leadingTerm('lex');
    if (leadingTerm === null || leadingTerm.coefficient < 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang leading-coefficient reconstruction', requirement: (0, errors_1.message)('polyReqPositiveLexLeadingSign') }));
    }
    const mainVariable = variables[0];
    const originalDegree = polynomial.degree(mainVariable);
    if (originalDegree === null || originalDegree === 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang configuration search', requirement: (0, errors_1.message)('polyReqPositiveDegreeMainVariable') }));
    }
    const leadingFactorization = factorWangLeadingCoefficient(polynomial, variables);
    if (leadingFactorization === null) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang configuration search', requirement: (0, errors_1.message)('polyReqPositiveDegreeMainVariable') }));
    }
    return {
        mainVariable,
        originalDegree,
        leadingFactorization,
    };
}
/**
 * Finds the nearest Wang configuration whose leading coefficients can be reconstructed
 * from the specialized factorization.
 */
function findWangConfiguration(polynomial, variables) {
    const { mainVariable, originalDegree, leadingFactorization, } = prepareWangConfigurationSearch(polynomial, variables);
    let radius = 0n;
    while (true) {
        casBudgetCheck();
        let found = null;
        visitEvaluationShell(variables.length - 1, radius, values => {
            const candidate = testWangConfigurationWithLeadingFactorization(polynomial, variables, values, leadingFactorization, mainVariable, originalDegree);
            if (candidate === null) {
                return false;
            }
            found = candidate;
            return true;
        });
        if (found !== null) {
            return found;
        }
        radius++;
    }
}
/** Finds the nearest bivariate Wang configuration. */
function findWangBivariateConfiguration(polynomial, variables) {
    if (variables.length !== 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Bivariate Wang configuration search', requirement: (0, errors_1.message)('polyReqExactlyTwoSelectedVariables') }));
    }
    return findWangConfiguration(polynomial, variables);
}
/**
 * Finds the nearest usable Wang/EEZ evaluation point in deterministic integer shells.
 *
 * The origin is tested first, followed by the boundary of [-1,1]^n, [-2,2]^n, and so
 * on. Search continues until a specialization preserves leading degree and is square-free.
 */
function findWangEvaluationPoint(polynomial, variables) {
    assertWangVariables(polynomial, variables);
    if (polynomial.isZero()) {
        throw new RangeError((0, errors_1.message)('polyWangCannotSpecializeZero'));
    }
    const mainVariable = variables[0];
    const originalDegree = polynomial.degree(mainVariable);
    if (originalDegree === null || originalDegree === 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Wang evaluation search', requirement: (0, errors_1.message)('polyReqPositiveDegreeMainVariable') }));
    }
    let radius = 0n;
    while (true) {
        casBudgetCheck();
        let found = null;
        visitEvaluationShell(variables.length - 1, radius, values => {
            const candidate = testWangSpecializationPrepared(polynomial, variables, values, mainVariable, originalDegree);
            if (candidate === null) {
                return false;
            }
            found = factorWangSpecialization(candidate, mainVariable);
            return true;
        });
        if (found !== null) {
            return found;
        }
        radius++;
    }
}

};
modules["algebra/polynomial/SparsePolynomialFactor"]=function(require,module,exports){
"use strict";
/**
 * @module sparsePolynomialFactor
 *
 * Exact polynomial factorization foundations over the integers.
 *
 * External references consulted for the mathematical algorithms:
 * - SymPy square-free decomposition (Yun's algorithm):
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/sqfreetools.py
 * - SymPy integer factorization pipeline (primitive/square-free decomposition before
 *   Zassenhaus factor splitting):
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/factortools.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.contentInVariable = contentInVariable;
exports.recombineLiftedUnivariateFactors = recombineLiftedUnivariateFactors;
exports.factorSquareFreeUnivariateInteger = factorSquareFreeUnivariateInteger;
exports.squareFreeFactorizationMultivariate = squareFreeFactorizationMultivariate;
exports.factorUnivariateInteger = factorUnivariateInteger;
exports.squareFreeFactorizationUnivariate = squareFreeFactorizationUnivariate;
const SparsePolynomial_1 = require("../../core/classes/polynomial/SparsePolynomial");
const errors_1 = require("../../core/errors");
const bigint_1 = require("../../core/functions/bigint/bigint");
const primeFactor_1 = require("../../core/functions/bigint/primeFactor");
const modularGcd_1 = require("./modularGcd");
const ModularSparsePolynomial_1 = require("./ModularSparsePolynomial");
const ModularSparsePolynomialFactor_1 = require("./ModularSparsePolynomialFactor");
const SparsePolynomialGcd_1 = require("./SparsePolynomialGcd");
const INITIAL_FACTOR_PRIME = 3n;
const MAX_FACTOR_PRIME_PROBES = 4;
function assertUnivariate(polynomial, variableIndex) {
    polynomial.degree(variableIndex);
    for (const term of polynomial.terms()) {
        for (let index = 0; index < polynomial.variableCount; index++) {
            if (index !== variableIndex && term.exponents[index] !== 0n) {
                throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Square-free univariate factorization', requirement: (0, errors_1.message)('polyReqAllOtherVariablesDormant') }));
            }
        }
    }
}
function findGoodFactorPrime(polynomial, variableIndex) {
    const leading = polynomial.leadingTerm('lex');
    if (leading === null) {
        throw new RangeError((0, errors_1.message)('polyIntegerFactorPrimeZero'));
    }
    const derivative = polynomial.derivative(variableIndex);
    let prime = (0, primeFactor_1.nextPrimeBig)(INITIAL_FACTOR_PRIME);
    let probes = 0;
    let selected;
    while (true) {
        casBudgetCheck();
        if (leading.coefficient % prime !== 0n) {
            const modularPolynomial = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, prime, polynomial.terms());
            const modularDerivative = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, prime, derivative.terms());
            const modularGcd = (0, modularGcd_1.modularGcdUnivariate)(modularPolynomial, modularDerivative, variableIndex);
            if (modularGcd.isConstant()) {
                const modularFactors = (0, ModularSparsePolynomialFactor_1.factorSquareFreeUnivariateFiniteField)(modularPolynomial, variableIndex);
                const candidate = { prime, modularPolynomial, modularFactors };
                probes++;
                // An irreducible image modulo a degree-preserving prime proves the
                // primitive integer polynomial irreducible, so no lifting is needed.
                if (modularFactors.length === 1) {
                    return candidate;
                }
                if (selected === undefined ||
                    modularFactors.length < selected.modularFactors.length) {
                    selected = candidate;
                }
                if (probes >= MAX_FACTOR_PRIME_PROBES) {
                    return selected;
                }
            }
        }
        prime = (0, primeFactor_1.nextPrimeBig)(prime + 1n);
    }
}
function assertFactorVariables(polynomial, variables) {
    if (variables.length === 0) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Sparse polynomial factorization', requirement: (0, errors_1.message)('polyReqAtLeastOneVariable') }));
    }
    const selected = new Set();
    let previous = -1;
    for (const variableIndex of variables) {
        polynomial.degree(variableIndex);
        if (selected.has(variableIndex)) {
            throw new RangeError((0, errors_1.message)('polyVariablesUnique', { operation: 'Sparse polynomial factorization' }));
        }
        if (variableIndex <= previous) {
            throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Sparse polynomial factorization' }));
        }
        selected.add(variableIndex);
        previous = variableIndex;
    }
    for (const activeVariable of polynomial.variables()) {
        if (!selected.has(activeVariable)) {
            throw new RangeError((0, errors_1.message)('polyVariablesIncludeActive', { operation: 'Sparse polynomial factorization' }));
        }
    }
}
function contentInVariable(polynomial, variableIndex, coefficientVariables) {
    const grouped = new Map();
    for (const term of polynomial.terms()) {
        const exponent = term.exponents[variableIndex];
        const exponents = [...term.exponents];
        exponents[variableIndex] = 0n;
        const group = grouped.get(exponent) ?? [];
        group.push({ coefficient: term.coefficient, exponents });
        grouped.set(exponent, group);
    }
    let content = null;
    for (const terms of grouped.values()) {
        const coefficient = new SparsePolynomial_1.SparsePolynomial(polynomial.variableCount, terms)
            .primitivePart()
            .normalizeLeadingSign('lex');
        if (content === null) {
            content = coefficient;
        }
        else if (coefficient.divideExact(content) !== null) {
            continue;
        }
        else if (content.divideExact(coefficient) !== null) {
            content = coefficient;
        }
        else {
            content = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(content, coefficient, coefficientVariables);
        }
        if (content.isConstant()) {
            return SparsePolynomial_1.SparsePolynomial.constant(polynomial.variableCount, 1n);
        }
    }
    return content ?? SparsePolynomial_1.SparsePolynomial.zero(polynomial.variableCount);
}
function mergeSquareFreeFactor(factors, factor, multiplicity) {
    if (factor.isConstant()) {
        return;
    }
    const normalized = factor.primitivePart().normalizeLeadingSign('lex');
    const current = factors.get(multiplicity);
    factors.set(multiplicity, current === undefined ? normalized : current.multiply(normalized));
}
function squareFreeFactorizationRecursive(polynomial, variables) {
    if (variables.length === 1) {
        return squareFreeFactorizationUnivariate(polynomial, variables[0]);
    }
    let content = polynomial.content();
    let primitive = polynomial.primitivePart();
    const leading = primitive.leadingTerm('lex');
    if (leading !== null && leading.coefficient < 0n) {
        primitive = primitive.negate();
        content = -content;
    }
    if (primitive.isConstant()) {
        return { content, factors: [] };
    }
    const mainVariable = variables[0];
    const coefficientVariables = variables.slice(1);
    const variableContent = contentInVariable(primitive, mainVariable, coefficientVariables);
    const variablePrimitive = divideExactOrThrow(primitive, variableContent);
    const factorsByMultiplicity = new Map();
    const mainDegree = variablePrimitive.degree(mainVariable);
    if (mainDegree !== null && mainDegree > 0n) {
        const derivative = variablePrimitive.derivative(mainVariable);
        let gcd = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(variablePrimitive, derivative, variables);
        let p = divideExactOrThrow(variablePrimitive, gcd);
        let q = divideExactOrThrow(derivative, gcd);
        let multiplicity = 1n;
        while (true) {
        casBudgetCheck();
            const difference = q.subtract(p.derivative(mainVariable));
            if (difference.isZero()) {
                mergeSquareFreeFactor(factorsByMultiplicity, p, multiplicity);
                break;
            }
            gcd = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(p, difference, variables);
            p = divideExactOrThrow(p, gcd);
            q = divideExactOrThrow(difference, gcd);
            mergeSquareFreeFactor(factorsByMultiplicity, gcd, multiplicity);
            multiplicity++;
        }
    }
    const contentFactorization = squareFreeFactorizationRecursive(variableContent, coefficientVariables);
    content *= contentFactorization.content;
    for (const entry of contentFactorization.factors) {
        mergeSquareFreeFactor(factorsByMultiplicity, entry.factor, entry.multiplicity);
    }
    const factors = [...factorsByMultiplicity.entries()]
        .sort(([leftMultiplicity], [rightMultiplicity]) => leftMultiplicity < rightMultiplicity
        ? -1
        : leftMultiplicity > rightMultiplicity
            ? 1
            : 0)
        .map(([multiplicity, squareFreeFactor]) => ({
        factor: squareFreeFactor,
        multiplicity,
    }));
    return { content, factors };
}
function divideExactOrThrow(dividend, divisor) {
    const quotient = dividend.divideExact(divisor);
    if (quotient === null) {
        throw new Error((0, errors_1.message)('polySquareFreeNonExactDivision'));
    }
    return quotient;
}
function zassenhausCoefficientBound(polynomial, variableIndex) {
    const degree = polynomial.degree(variableIndex);
    if (degree === null || degree === 0n) {
        return polynomial.maxAbsoluteCoefficient();
    }
    const leading = polynomial.leadingTerm('lex');
    if (leading === null) {
        return 0n;
    }
    const leadingAbsolute = leading.coefficient < 0n ? -leading.coefficient : leading.coefficient;
    const maxCoefficient = polynomial.maxAbsoluteCoefficient();
    const root = (0, bigint_1.ceilIntegerSquareRoot)(degree + 1n);
    return root * (1n << degree) * maxCoefficient * leadingAbsolute;
}
function modularProduct(factors, indices, variableCount, modulus) {
    let product = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(variableCount, modulus, 1n);
    for (const index of indices) {
        product = product.multiply(factors[index]);
    }
    return product;
}
function combinations(values, size) {
    const result = [];
    const selected = [];
    function choose(start) {
        if (selected.length === size) {
            result.push([...selected]);
            return;
        }
        const remainingNeeded = size - selected.length;
        for (let index = start; index <= values.length - remainingNeeded; index++) {
            selected.push(values[index]);
            choose(index + 1);
            selected.pop();
        }
    }
    choose(0);
    return result;
}
/**
 * Recombines Hensel-lifted modular factors into exact integer factors.
 *
 * The modulus must exceed twice the Zassenhaus coefficient bound so symmetric
 * representatives determine integer factor coefficients uniquely. Candidate subsets
 * are accepted only after exact division in Z[x].
 */
function recombineLiftedUnivariateFactors(polynomial, liftedFactors, variableIndex) {
    assertUnivariate(polynomial, variableIndex);
    if (polynomial.isZero()) {
        throw new RangeError((0, errors_1.message)('polyCannotFactorZeroPolynomial', { operation: 'Zassenhaus recombination' }));
    }
    if (polynomial.content() !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqPrimitiveIntegerPolynomial') }));
    }
    const leading = polynomial.leadingTerm('lex');
    if (leading === null || leading.coefficient <= 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqPositiveLeadingCoefficient') }));
    }
    if (liftedFactors.length === 0) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqLiftedModularFactors') }));
    }
    const modulus = liftedFactors[0].modulus;
    for (const factor of liftedFactors) {
        if (factor.variableCount !== polynomial.variableCount ||
            factor.modulus !== modulus) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqLiftedFactorsSameRing') }));
        }
        for (const activeVariable of factor.variables()) {
            if (activeVariable !== variableIndex) {
                throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqUnivariateLiftedFactors') }));
            }
        }
        if (!factor.equals(factor.monic())) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqMonicLiftedFactors') }));
        }
    }
    const coefficientBound = zassenhausCoefficientBound(polynomial, variableIndex);
    if (modulus <= 2n * coefficientBound) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Zassenhaus recombination', requirement: (0, errors_1.message)('polyReqLargerHenselModulus') }));
    }
    const modularInput = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, modulus, polynomial.terms());
    const allIndices = liftedFactors.map((_factor, index) => index);
    const modularReconstruction = modularProduct(liftedFactors, allIndices, polynomial.variableCount, modulus).scale(leading.coefficient);
    if (!modularReconstruction.equals(modularInput)) {
        throw new RangeError((0, errors_1.message)('polyZassenhausFactorsReconstructModuloLift'));
    }
    const factors = [];
    let remainingPolynomial = polynomial;
    let remainingIndices = [...allIndices];
    let subsetSize = 1;
    while (2 * subsetSize <= remainingIndices.length) {
        let accepted = false;
        for (const subset of combinations(remainingIndices, subsetSize)) {
            const remainingLeading = remainingPolynomial.leadingTerm('lex');
            if (remainingLeading === null) {
                throw new Error((0, errors_1.message)('polyZassenhausLostRemaining'));
            }
            const modularCandidate = modularProduct(liftedFactors, subset, polynomial.variableCount, modulus).scale(remainingLeading.coefficient);
            let candidate = modularCandidate.toSymmetricIntegerPolynomial().primitivePart();
            candidate = candidate.normalizeLeadingSign('lex');
            if (candidate.isConstant()) {
                continue;
            }
            const quotient = remainingPolynomial.divideExact(candidate);
            if (quotient === null) {
                continue;
            }
            factors.push(candidate);
            remainingPolynomial = quotient.primitivePart().normalizeLeadingSign('lex');
            const selected = new Set(subset);
            remainingIndices = remainingIndices.filter(index => !selected.has(index));
            subsetSize = 1;
            accepted = true;
            break;
        }
        if (!accepted) {
            subsetSize++;
        }
    }
    if (!remainingPolynomial.isConstant()) {
        factors.push(remainingPolynomial);
    }
    let reconstruction = SparsePolynomial_1.SparsePolynomial.constant(polynomial.variableCount, 1n);
    for (const factor of factors) {
        reconstruction = reconstruction.multiply(factor);
    }
    if (!reconstruction.equals(polynomial)) {
        throw new Error((0, errors_1.message)('polyZassenhausReconstructionFailed'));
    }
    return factors;
}
/**
 * Factors a primitive square-free univariate polynomial into irreducibles over Z.
 *
 * A prime preserving degree and square-freeness is selected deterministically.
 * Berlekamp factorization supplies the modular factors; reducible modular images are
 * lifted until the coefficient reconstruction bound is exceeded and then recombined
 * by exact integer division.
 */
function factorSquareFreeUnivariateInteger(polynomial, variableIndex) {
    assertUnivariate(polynomial, variableIndex);
    if (polynomial.isZero()) {
        throw new RangeError((0, errors_1.message)('polyCannotFactorZeroPolynomial', { operation: 'Integer factorization' }));
    }
    if (polynomial.content() !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Integer square-free factorization', requirement: (0, errors_1.message)('polyReqPrimitivePolynomial') }));
    }
    const leading = polynomial.leadingTerm('lex');
    if (leading === null) {
        throw new Error((0, errors_1.message)('polyIntegerFactorLostLeading'));
    }
    const input = leading.coefficient < 0n ? polynomial.negate() : polynomial;
    if (input.isConstant()) {
        return [];
    }
    const degree = input.degree(variableIndex);
    if (degree === 1n) {
        return [input];
    }
    const selected = findGoodFactorPrime(input, variableIndex);
    if (selected.modularFactors.length === 1) {
        return [input];
    }
    const coefficientBound = zassenhausCoefficientBound(input, variableIndex);
    let targetModulus = selected.prime;
    while (targetModulus <= 2n * coefficientBound) {
        targetModulus *= targetModulus;
    }
    const lifted = (0, ModularSparsePolynomialFactor_1.henselLiftUnivariateFactors)(input, selected.prime, selected.modularFactors, variableIndex, targetModulus);
    return recombineLiftedUnivariateFactors(input, lifted, variableIndex);
}
/**
 * Computes square-free decomposition of an integer polynomial in one or more variables.
 *
 * The polynomial is treated recursively as univariate in the first selected variable.
 * Coefficient content is decomposed over the remaining variables and factors with the
 * same multiplicity are multiplied together.
 */
function squareFreeFactorizationMultivariate(polynomial, variables) {
    assertFactorVariables(polynomial, variables);
    if (polynomial.isZero()) {
        return { content: 0n, factors: [] };
    }
    return squareFreeFactorizationRecursive(polynomial, variables);
}
/**
 * Completely factors a univariate integer polynomial.
 *
 * Content and global sign are returned separately. Irreducible primitive factors are
 * paired with the multiplicities obtained from square-free decomposition.
 */
function factorUnivariateInteger(polynomial, variableIndex) {
    const squareFree = squareFreeFactorizationUnivariate(polynomial, variableIndex);
    if (squareFree.content === 0n || squareFree.factors.length === 0) {
        return squareFree;
    }
    const factors = [];
    for (const entry of squareFree.factors) {
        const irreducible = factorSquareFreeUnivariateInteger(entry.factor, variableIndex);
        for (const factor of irreducible) {
            factors.push({
                factor,
                multiplicity: entry.multiplicity,
            });
        }
    }
    return {
        content: squareFree.content,
        factors,
    };
}
/**
 * Computes the square-free decomposition of a univariate integer polynomial.
 *
 * The returned content carries the sign needed to reconstruct the input. Each
 * nonconstant factor is primitive with positive leading coefficient. A common power
 * of the selected variable is extracted before Yun's loop so sparse monomials such
 * as x^N do not require N iterations.
 */
function squareFreeFactorizationUnivariate(polynomial, variableIndex) {
    assertUnivariate(polynomial, variableIndex);
    if (polynomial.isZero()) {
        return { content: 0n, factors: [] };
    }
    let content = polynomial.content();
    let primitive = polynomial.primitivePart();
    const leading = primitive.leadingTerm('lex');
    if (leading !== null && leading.coefficient < 0n) {
        primitive = primitive.negate();
        content = -content;
    }
    if (primitive.isConstant()) {
        return { content, factors: [] };
    }
    const factors = [];
    let minimumExponent = null;
    for (const term of primitive.terms()) {
        const exponent = term.exponents[variableIndex];
        if (minimumExponent === null || exponent < minimumExponent) {
            minimumExponent = exponent;
        }
    }
    if (minimumExponent !== null && minimumExponent > 0n) {
        const exponents = new Array(primitive.variableCount).fill(0n);
        exponents[variableIndex] = minimumExponent;
        const monomial = SparsePolynomial_1.SparsePolynomial.monomial(primitive.variableCount, 1n, exponents);
        primitive = divideExactOrThrow(primitive, monomial);
        factors.push({
            factor: SparsePolynomial_1.SparsePolynomial.variable(primitive.variableCount, variableIndex),
            multiplicity: minimumExponent,
        });
    }
    if (primitive.isConstant()) {
        return { content, factors };
    }
    const derivative = primitive.derivative(variableIndex);
    const variableIndices = [variableIndex];
    let gcd = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(primitive, derivative, variableIndices);
    let p = divideExactOrThrow(primitive, gcd);
    let q = divideExactOrThrow(derivative, gcd);
    let multiplicity = 1n;
    while (true) {
        casBudgetCheck();
        const difference = q.subtract(p.derivative(variableIndex));
        if (difference.isZero()) {
            if (!p.isConstant()) {
                factors.push({
                    factor: p.normalizeLeadingSign('lex'),
                    multiplicity,
                });
            }
            break;
        }
        gcd = (0, SparsePolynomialGcd_1.sparsePolynomialGcd)(p, difference, variableIndices);
        p = divideExactOrThrow(p, gcd);
        q = divideExactOrThrow(difference, gcd);
        if (!gcd.isConstant()) {
            factors.push({
                factor: gcd.normalizeLeadingSign('lex'),
                multiplicity,
            });
        }
        multiplicity++;
    }
    factors.sort((left, right) => left.multiplicity < right.multiplicity
        ? -1
        : left.multiplicity > right.multiplicity
            ? 1
            : 0);
    return { content, factors };
}

};
modules["algebra/polynomial/SparsePolynomialGcd"]=function(require,module,exports){
"use strict";
/**
 * @module sparsePolynomialGcd
 *
 * Integer GCD reconstruction for the exact sparse polynomial representation.
 *
 * External references consulted:
 * - SymPy multivariate Chinese remainder reconstruction and modular GCD loop:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/zippel.py
 * - SymPy Chinese Remainder Theorem implementation:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/ntheory/modular.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.combinePolynomialCongruences = combinePolynomialCongruences;
exports.sparsePolynomialGcd = sparsePolynomialGcd;
const SparsePolynomial_1 = require("../../core/classes/polynomial/SparsePolynomial");
const errors_1 = require("../../core/errors");
const bigint_1 = require("../../core/functions/bigint/bigint");
const primeFactor_1 = require("../../core/functions/bigint/primeFactor");
const modularGcd_1 = require("./modularGcd");
const ModularSparsePolynomial_1 = require("./ModularSparsePolynomial");
// SymPy's integer Zippel loop starts near 10^9 because larger fields sharply reduce
// failed evaluations during sparse interpolation.
const INITIAL_GCD_PRIME = 1000000007n;
/** Maps a residue to the symmetric representative for one positive modulus. */
function symmetricResidue(value, modulus) {
    const residue = (0, bigint_1.mod)(value, modulus);
    return residue * 2n > modulus ? residue - modulus : residue;
}
/** Looks up a term coefficient without interpreting the monomial key. */
function termMap(polynomial) {
    const terms = new Map();
    for (const term of polynomial.terms()) {
        terms.set(term.exponents.join(','), term);
    }
    return terms;
}
/**
 * Combines two sparse polynomial congruences with coprime positive moduli.
 *
 * Missing monomials are treated as zero residues. Exponent vectors are retained
 * from the source term and never reconstructed from the map key.
 */
function combinePolynomialCongruences(left, leftModulus, right, rightModulus) {
    if (left.variableCount !== right.variableCount) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Polynomial CRT', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
    }
    if (leftModulus <= 0n || rightModulus <= 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Polynomial CRT', requirement: (0, errors_1.message)('polyReqPositiveModuli') }));
    }
    if ((0, bigint_1.GCD)(leftModulus, rightModulus) !== 1n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Polynomial CRT', requirement: (0, errors_1.message)('polyReqCoprimeModuli') }));
    }
    const inverse = (0, bigint_1.invMod)(leftModulus, rightModulus);
    if (inverse < 0n) {
        throw new RangeError((0, errors_1.message)('polyCrtModuliInvertible'));
    }
    const leftTerms = termMap(left);
    const rightTerms = termMap(right);
    const keys = new Set([...leftTerms.keys(), ...rightTerms.keys()]);
    const modulus = leftModulus * rightModulus;
    const terms = [];
    for (const key of keys) {
        const leftTerm = leftTerms.get(key);
        const rightTerm = rightTerms.get(key);
        const leftCoefficient = leftTerm?.coefficient ?? 0n;
        const rightCoefficient = rightTerm?.coefficient ?? 0n;
        const correction = (0, bigint_1.mod)((rightCoefficient - leftCoefficient) * inverse, rightModulus);
        const coefficient = symmetricResidue(leftCoefficient + leftModulus * correction, modulus);
        if (coefficient !== 0n) {
            terms.push({
                coefficient,
                exponents: leftTerm?.exponents ?? rightTerm?.exponents ?? [],
            });
        }
    }
    return {
        polynomial: new SparsePolynomial_1.SparsePolynomial(left.variableCount, terms),
        modulus,
    };
}
/**
 * Returns the integer leading coefficient after making one variable primary.
 *
 * Ties follow the remaining ring coordinates in their existing order, matching a
 * lexicographic view with variableIndex moved to the front.
 */
function leadingCoefficientForVariable(polynomial, variableIndex) {
    polynomial.degree(variableIndex);
    const terms = polynomial.terms();
    if (terms.length === 0) {
        return 0n;
    }
    let leading = terms[0];
    for (let termIndex = 1; termIndex < terms.length; termIndex++) {
        const term = terms[termIndex];
        if (term.exponents[variableIndex] !== leading.exponents[variableIndex]) {
            if (term.exponents[variableIndex] > leading.exponents[variableIndex]) {
                leading = term;
            }
            continue;
        }
        for (let index = 0; index < polynomial.variableCount; index++) {
            if (index === variableIndex) {
                continue;
            }
            if (term.exponents[index] === leading.exponents[index]) {
                continue;
            }
            if (term.exponents[index] > leading.exponents[index]) {
                leading = term;
            }
            break;
        }
    }
    return leading.coefficient;
}
/** Converts an integer sparse polynomial to canonical residues modulo prime. */
function reduceModulo(polynomial, prime) {
    return new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, prime, polynomial.terms());
}
/** Returns the selected-variable degree vector used to compare modular images. */
function degreeVector(polynomial, variables) {
    return variables.map(variableIndex => polynomial.degree(variableIndex) ?? 0n);
}
/**
 * Computes the integer GCD of two sparse polynomials by modular images and CRT.
 *
 * variables must follow polynomial-ring order and contain every active variable.
 * The returned GCD has positive leading coefficient under lexicographic order.
 */
function sparsePolynomialGcd(left, right, variables) {
    if (left.variableCount !== right.variableCount) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Sparse polynomial GCD', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
    }
    if (variables.length === 0) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Sparse polynomial GCD', requirement: (0, errors_1.message)('polyReqAtLeastOneVariable') }));
    }
    const selected = new Set();
    let previousVariable = -1;
    for (const variableIndex of variables) {
        left.degree(variableIndex);
        right.degree(variableIndex);
        if (selected.has(variableIndex)) {
            throw new RangeError((0, errors_1.message)('polyVariablesUnique', { operation: 'Sparse polynomial GCD' }));
        }
        if (variableIndex <= previousVariable) {
            throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Sparse polynomial GCD' }));
        }
        selected.add(variableIndex);
        previousVariable = variableIndex;
    }
    for (const activeVariable of [...left.variables(), ...right.variables()]) {
        if (!selected.has(activeVariable)) {
            throw new RangeError((0, errors_1.message)('polyInputsSelectedOnly', { operation: 'Sparse polynomial GCD' }));
        }
    }
    if (left.isZero()) {
        return right.isZero() ? right : right.normalizeLeadingSign('lex');
    }
    if (right.isZero()) {
        return left.normalizeLeadingSign('lex');
    }
    const leftContent = left.content();
    const rightContent = right.content();
    const commonContent = (0, bigint_1.GCD)(leftContent, rightContent);
    const primitiveLeft = left.primitivePart().normalizeLeadingSign('lex');
    const primitiveRight = right.primitivePart().normalizeLeadingSign('lex');
    const leftLeading = primitiveLeft.leadingTerm('lex');
    const rightLeading = primitiveRight.leadingTerm('lex');
    if (leftLeading === null || rightLeading === null) {
        return SparsePolynomial_1.SparsePolynomial.zero(left.variableCount);
    }
    if (primitiveLeft.isConstant() || primitiveRight.isConstant()) {
        return SparsePolynomial_1.SparsePolynomial.constant(left.variableCount, commonContent);
    }
    if (primitiveLeft.equals(primitiveRight)) {
        return primitiveLeft.scale(commonContent);
    }
    if (primitiveLeft.divideExact(primitiveRight) !== null) {
        return primitiveRight.scale(commonContent);
    }
    if (primitiveRight.divideExact(primitiveLeft) !== null) {
        return primitiveLeft.scale(commonContent);
    }
    const leadingGcd = (0, bigint_1.GCD)(leftLeading.coefficient, rightLeading.coefficient);
    const badPrimeFactors = variables.map(variableIndex => (0, bigint_1.GCD)(leadingCoefficientForVariable(primitiveLeft, variableIndex), leadingCoefficientForVariable(primitiveRight, variableIndex)));
    let prime = (0, primeFactor_1.nextPrimeBig)(INITIAL_GCD_PRIME);
    let accumulated = null;
    let accumulatedModulus = 1n;
    let expectedDegrees = null;
    while (true) {
        casBudgetCheck();
        while (badPrimeFactors.some(factor => factor % prime === 0n)) {
            prime = (0, primeFactor_1.nextPrimeBig)(prime + 1n);
        }
        const modularLeft = reduceModulo(primitiveLeft, prime);
        const modularRight = reduceModulo(primitiveRight, prime);
        const modularGcd = (0, modularGcd_1.modularGcdFiniteField)(modularLeft, modularRight, variables);
        if (modularGcd === null) {
            prime = (0, primeFactor_1.nextPrimeBig)(prime + 1n);
            continue;
        }
        if (modularGcd.isConstant() && !modularGcd.isZero()) {
            return SparsePolynomial_1.SparsePolynomial.constant(left.variableCount, commonContent);
        }
        const imageDegrees = degreeVector(modularGcd, variables);
        if (expectedDegrees !== null) {
            let degreeComparison = 0;
            for (let index = 0; index < imageDegrees.length; index++) {
                if (imageDegrees[index] < expectedDegrees[index]) {
                    degreeComparison = -1;
                    break;
                }
                if (imageDegrees[index] > expectedDegrees[index]) {
                    degreeComparison = 1;
                    break;
                }
            }
            if (degreeComparison > 0) {
                prime = (0, primeFactor_1.nextPrimeBig)(prime + 1n);
                continue;
            }
            if (degreeComparison < 0) {
                expectedDegrees = imageDegrees;
                accumulated = null;
                accumulatedModulus = 1n;
            }
        }
        else {
            expectedDegrees = imageDegrees;
        }
        const scaledImage = modularGcd.scale(leadingGcd);
        const liftedImage = scaledImage.toSymmetricIntegerPolynomial();
        let reconstructed;
        let reconstructedModulus;
        if (accumulated === null) {
            reconstructed = liftedImage;
            reconstructedModulus = prime;
        }
        else {
            const combined = combinePolynomialCongruences(accumulated, accumulatedModulus, liftedImage, prime);
            reconstructed = combined.polynomial;
            reconstructedModulus = combined.modulus;
        }
        const stable = accumulated !== null && reconstructed.equals(accumulated);
        accumulated = reconstructed;
        accumulatedModulus = reconstructedModulus;
        if (stable) {
            const primitiveCandidate = reconstructed
                .primitivePart()
                .normalizeLeadingSign('lex');
            const candidate = primitiveCandidate.scale(commonContent);
            if (left.divideExact(candidate) !== null && right.divideExact(candidate) !== null) {
                return candidate;
            }
        }
        prime = (0, primeFactor_1.nextPrimeBig)(prime + 1n);
    }
}

};
modules["algebra/polynomial/ModularSparsePolynomial"]=function(require,module,exports){
"use strict";
/**
 * @module modularSparsePolynomial
 *
 * Sparse polynomial arithmetic over integer residue classes.
 *
 * External references consulted for modular coefficient arithmetic and sparse
 * polynomial division:
 * - SymPy finite-field polynomial arithmetic:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/galoistools.py
 * - SymPy sparse polynomial substitution, evaluation, and term division:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/sparsetools.py
 * - SymPy variable-relative leading coefficient operations used by modular GCD:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/zippel.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ModularSparsePolynomial = void 0;
const SparsePolynomial_1 = require("../../core/classes/polynomial/SparsePolynomial");
const errors_1 = require("../../core/errors");
const bigint_1 = require("../../core/functions/bigint/bigint");
/**
 * Sparse polynomial whose coefficients are stored as canonical residues modulo
 * a positive integer greater than one.
 *
 * The modulus is part of the polynomial ring. Prime moduli provide finite-field
 * arithmetic; composite moduli remain valid for addition and multiplication but
 * may not provide multiplicative inverses required by monic normalization or
 * polynomial division.
 */
class ModularSparsePolynomial {
    constructor(variableCount, modulus, terms = []) {
        if (modulus <= 1n) {
            throw new RangeError((0, errors_1.message)('polyModulusGreaterThanOne'));
        }
        let combined;
        if (terms instanceof SparsePolynomial_1.SparsePolynomial) {
            if (terms.variableCount !== variableCount) {
                throw new RangeError((0, errors_1.message)('polyVariableCountMatch'));
            }
            combined = terms;
        }
        else {
            combined = new SparsePolynomial_1.SparsePolynomial(variableCount, terms);
        }
        this.modulus = modulus;
        this.polynomial = combined.mapCoefficients(coefficient => (0, bigint_1.mod)(coefficient, modulus));
        this.variableCount = this.polynomial.variableCount;
    }
    /** Creates a constant polynomial in the selected residue ring. */
    static constant(variableCount, modulus, coefficient) {
        return new ModularSparsePolynomial(variableCount, modulus, [
            {
                coefficient,
                exponents: new Array(variableCount).fill(0n),
            },
        ]);
    }
    /**
     * Reconstructs a polynomial in one variable from specialized images.
     *
     * Each sample polynomial must be independent of the interpolated variable.
     * Evaluation values are compared as residues in the shared coefficient ring.
     */
    static interpolateVariable(variableIndex, samples) {
        if (samples.length === 0) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'ModularSparsePolynomial interpolation', requirement: (0, errors_1.message)('polyReqAtLeastOneSample') }));
        }
        const first = samples[0].polynomial;
        first.degree(variableIndex);
        const x = ModularSparsePolynomial.variable(first.variableCount, first.modulus, variableIndex);
        let result = ModularSparsePolynomial.zero(first.variableCount, first.modulus);
        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i];
            first.assertCompatible(sample.polynomial);
            const sampleDegree = sample.polynomial.degree(variableIndex);
            if (sampleDegree !== null && sampleDegree !== 0n) {
                throw new RangeError((0, errors_1.message)('polyInterpolationCannotDepend'));
            }
            const value = (0, bigint_1.mod)(sample.value, first.modulus);
            let denominator = 1n;
            let basis = ModularSparsePolynomial.constant(first.variableCount, first.modulus, 1n);
            for (let j = 0; j < samples.length; j++) {
                if (i === j) {
                    continue;
                }
                const otherValue = (0, bigint_1.mod)(samples[j].value, first.modulus);
                basis = basis.multiply(x.subtract(ModularSparsePolynomial.constant(first.variableCount, first.modulus, otherValue)));
                denominator = (0, bigint_1.mod)(denominator * (0, bigint_1.mod)(value - otherValue, first.modulus), first.modulus);
            }
            const inverse = (0, bigint_1.invMod)(denominator, first.modulus);
            if (inverse < 0n) {
                throw new RangeError((0, errors_1.message)('polyInterpolationInvertibleDifferences'));
            }
            result = result.add(sample.polynomial.multiply(basis.scale(inverse)));
        }
        return result;
    }
    /** Creates a one-term polynomial in the selected residue ring. */
    static monomial(variableCount, modulus, coefficient, exponents) {
        return new ModularSparsePolynomial(variableCount, modulus, [{ coefficient, exponents }]);
    }
    /** Creates one polynomial variable with coefficient one. */
    static variable(variableCount, modulus, variableIndex) {
        const variable = SparsePolynomial_1.SparsePolynomial.variable(variableCount, variableIndex);
        return new ModularSparsePolynomial(variableCount, modulus, variable);
    }
    /** Creates the zero polynomial in the selected residue ring. */
    static zero(variableCount, modulus) {
        return new ModularSparsePolynomial(variableCount, modulus);
    }
    /** Verifies that another polynomial belongs to the same residue ring. */
    assertCompatible(other) {
        if (this.variableCount !== other.variableCount || this.modulus !== other.modulus) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'ModularSparsePolynomial operations', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
        }
    }
    /** Adds another polynomial in the same residue ring. */
    add(other) {
        this.assertCompatible(other);
        const sum = this.polynomial.add(other.polynomial);
        return new ModularSparsePolynomial(this.variableCount, this.modulus, sum);
    }
    /** Returns the canonical residue coefficient of one monomial. */
    coefficient(exponents) {
        return this.polynomial.coefficient(exponents);
    }
    /**
     * Returns the coefficient polynomial at one exact power of a selected variable.
     *
     * The selected variable remains in the ring with exponent zero.
     */
    coefficientIn(variableIndex, exponent) {
        this.polynomial.degree(variableIndex);
        if (exponent < 0n) {
            throw new RangeError((0, errors_1.message)('polyCoefficientExponentNonnegative'));
        }
        const terms = [];
        for (const term of this.polynomial.terms()) {
            if (term.exponents[variableIndex] === exponent) {
                const exponents = [...term.exponents];
                exponents[variableIndex] = 0n;
                terms.push({
                    coefficient: term.coefficient,
                    exponents,
                });
            }
        }
        return new ModularSparsePolynomial(this.variableCount, this.modulus, terms);
    }
    /** Returns the canonical residue of the constant coefficient. */
    constantTerm() {
        return this.polynomial.constantTerm();
    }
    /** Returns the degree in one variable, or null for the zero polynomial. */
    degree(variableIndex) {
        return this.polynomial.degree(variableIndex);
    }
    /**
     * Differentiates with respect to one ring variable in the residue ring.
     */
    derivative(variableIndex) {
        this.polynomial.degree(variableIndex);
        const terms = [];
        for (const term of this.polynomial.terms()) {
            const exponent = term.exponents[variableIndex];
            if (exponent === 0n) {
                continue;
            }
            const exponents = [...term.exponents];
            exponents[variableIndex] = exponent - 1n;
            terms.push({
                coefficient: (0, bigint_1.mod)(term.coefficient * exponent, this.modulus),
                exponents,
            });
        }
        return new ModularSparsePolynomial(this.variableCount, this.modulus, terms);
    }
    /**
     * Divides by one polynomial when the remainder is zero.
     */
    divideExact(divisor, order = 'lex') {
        const division = this.divideWithRemainder(divisor, order);
        return division.remainder.isZero() ? division.quotient : null;
    }
    /**
     * Divides by one polynomial and returns quotient and remainder.
     *
     * The leading coefficient of the divisor must be invertible modulo the ring
     * modulus.
     */
    divideWithRemainder(divisor, order = 'lex') {
        this.assertCompatible(divisor);
        const divisorLeading = divisor.leadingTerm(order);
        if (divisorLeading === null) {
            throw new RangeError((0, errors_1.message)('polyCannotDivideZero'));
        }
        const divisorLeadingInverse = (0, bigint_1.invMod)(divisorLeading.coefficient, this.modulus);
        if (divisorLeadingInverse < 0n) {
            throw new RangeError((0, errors_1.message)('polyDivisorLeadingCoefficientInvertible'));
        }
        let quotient = ModularSparsePolynomial.zero(this.variableCount, this.modulus);
        let remainder = ModularSparsePolynomial.zero(this.variableCount, this.modulus);
        let working = new ModularSparsePolynomial(this.variableCount, this.modulus, this.polynomial);
        while (!working.isZero()) {
            const leading = working.leadingTerm(order);
            if (leading === null) {
                break;
            }
            let divides = true;
            const exponents = new Array(this.variableCount);
            for (let variableIndex = 0; variableIndex < this.variableCount; variableIndex++) {
                if (leading.exponents[variableIndex] < divisorLeading.exponents[variableIndex]) {
                    divides = false;
                    break;
                }
                exponents[variableIndex] =
                    leading.exponents[variableIndex] - divisorLeading.exponents[variableIndex];
            }
            if (divides) {
                const coefficient = (0, bigint_1.mod)(leading.coefficient * divisorLeadingInverse, this.modulus);
                const quotientTerm = ModularSparsePolynomial.monomial(this.variableCount, this.modulus, coefficient, exponents);
                quotient = quotient.add(quotientTerm);
                working = working.subtract(divisor.multiply(quotientTerm));
            }
            else {
                const remainderTerm = ModularSparsePolynomial.monomial(this.variableCount, this.modulus, leading.coefficient, leading.exponents);
                remainder = remainder.add(remainderTerm);
                working = working.subtract(remainderTerm);
            }
        }
        return { quotient, remainder };
    }
    /** Returns whether two polynomials have the same residue ring and exact terms. */
    equals(other) {
        return this.modulus === other.modulus && this.polynomial.equals(other.polynomial);
    }
    /**
     * Substitutes one variable with a residue while keeping the same ring dimension.
     *
     * Terms that become like terms after substitution are combined by the sparse
     * polynomial constructor.
     */
    evaluateVariable(variableIndex, value) {
        this.polynomial.degree(variableIndex);
        const residue = (0, bigint_1.mod)(value, this.modulus);
        const terms = [];
        for (const term of this.polynomial.terms()) {
            const exponent = term.exponents[variableIndex];
            const exponents = [...term.exponents];
            exponents[variableIndex] = 0n;
            terms.push({
                coefficient: (0, bigint_1.mod)(term.coefficient * (0, bigint_1.powMod)(residue, exponent, this.modulus), this.modulus),
                exponents,
            });
        }
        return new ModularSparsePolynomial(this.variableCount, this.modulus, terms);
    }
    /** Returns whether the polynomial is zero or contains only a constant term. */
    isConstant() {
        return this.polynomial.isConstant();
    }
    /** Returns whether the polynomial has no nonzero terms. */
    isZero() {
        return this.polynomial.isZero();
    }
    /**
     * Returns the coefficient polynomial of the highest power of one variable.
     *
     * The selected variable remains in the ring with exponent zero so recursive
     * algorithms can preserve a stable variable layout.
     */
    leadingCoefficientIn(variableIndex) {
        const degree = this.degree(variableIndex);
        return degree === null ? null : this.coefficientIn(variableIndex, degree);
    }
    /** Returns the leading term under the requested monomial order. */
    leadingTerm(order = 'lex') {
        return this.polynomial.leadingTerm(order);
    }
    /**
     * Scales the polynomial so its leading coefficient is one.
     *
     * Zero remains zero. A nonzero leading coefficient that has no inverse modulo
     * the modulus is rejected.
     */
    monic(order = 'lex') {
        const leading = this.leadingTerm(order);
        if (leading === null || leading.coefficient === 1n) {
            return this;
        }
        const inverse = (0, bigint_1.invMod)(leading.coefficient, this.modulus);
        if (inverse < 0n) {
            throw new RangeError((0, errors_1.message)('polyLeadingCoefficientInvertible'));
        }
        return this.scale(inverse);
    }
    /** Multiplies by another polynomial in the same residue ring. */
    multiply(other) {
        this.assertCompatible(other);
        const product = this.polynomial.multiply(other.polynomial);
        return new ModularSparsePolynomial(this.variableCount, this.modulus, product);
    }
    /** Returns the additive inverse in the residue ring. */
    negate() {
        const negated = this.polynomial.negate();
        return new ModularSparsePolynomial(this.variableCount, this.modulus, negated);
    }
    /** Raises the polynomial to a non-negative exact integer power. */
    pow(exponent) {
        if (exponent < 0n) {
            throw new RangeError((0, errors_1.message)('polyExponentNonnegative'));
        }
        let result = ModularSparsePolynomial.constant(this.variableCount, this.modulus, 1n);
        let power = exponent;
        if (power === 0n) {
            return result;
        }
        if ((power & 1n) === 1n) {
            result = result.multiply(this);
        }
        power >>= 1n;
        if (power === 0n) {
            return result;
        }
        let factor = this.multiply(this);
        while (power > 0n) {
            if ((power & 1n) === 1n) {
                result = result.multiply(factor);
            }
            power >>= 1n;
            if (power > 0n) {
                factor = factor.multiply(factor);
            }
        }
        return result;
    }
    /** Multiplies every coefficient by one residue-ring scalar. */
    scale(scalar) {
        const scaled = this.polynomial.scale((0, bigint_1.mod)(scalar, this.modulus));
        return new ModularSparsePolynomial(this.variableCount, this.modulus, scaled);
    }
    /** Subtracts another polynomial in the same residue ring. */
    subtract(other) {
        this.assertCompatible(other);
        const difference = this.polynomial.subtract(other.polynomial);
        return new ModularSparsePolynomial(this.variableCount, this.modulus, difference);
    }
    /** Number of stored nonzero terms. */
    get termCount() {
        return this.polynomial.termCount;
    }
    /** Returns the stored canonical residue terms. */
    terms() {
        return this.polynomial.terms();
    }
    /** Converts canonical residues to centered integer representatives. */
    toSymmetricIntegerPolynomial() {
        const half = this.modulus / 2n;
        return new SparsePolynomial_1.SparsePolynomial(this.variableCount, this.terms().map(term => ({
            coefficient: term.coefficient > half
                ? term.coefficient - this.modulus
                : term.coefficient,
            exponents: term.exponents,
        })));
    }
    /** Returns the largest total degree, or null for the zero polynomial. */
    totalDegree() {
        return this.polynomial.totalDegree();
    }
    /** Returns the variable indices that occur with a positive exponent. */
    variables() {
        return this.polynomial.variables();
    }
}
exports.ModularSparsePolynomial = ModularSparsePolynomial;

};
modules["algebra/polynomial/modularGcd"]=function(require,module,exports){
"use strict";
/**
 * @module modularGcd
 *
 * Greatest common divisors for sparse polynomials over a modular field.
 *
 * External references consulted for finite-field Euclidean GCD and recursive
 * evaluation/interpolation:
 * - SymPy finite-field polynomial GCD:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/galoistools.py
 * - SymPy sparse univariate modular GCD bridge:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/zippel.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.modularGcdUnivariate = modularGcdUnivariate;
exports.modularGcdFiniteField = modularGcdFiniteField;
exports.modularGcdBivariate = modularGcdBivariate;
exports.modularGcdTrivariate = modularGcdTrivariate;
const errors_1 = require("../../core/errors");
const primeFactor_1 = require("../../core/functions/bigint/primeFactor");
const ModularSparsePolynomial_1 = require("./ModularSparsePolynomial");
// SymPy gives failed sparse interpolation data a small number of fresh attempts
// before abandoning the current modular image.
const MODULAR_GCD_RECONSTRUCTION_ATTEMPTS = 3n;
/**
 * Computes polynomial content in one selected coefficient variable.
 *
 * Terms are grouped by their complete exponent vector outside coefficientVariable.
 * The grouping key is used only for identity and is never parsed back into exponents.
 */
function modularContentInVariable(polynomial, coefficientVariable) {
    polynomial.degree(coefficientVariable);
    if (polynomial.isZero()) {
        return ModularSparsePolynomial_1.ModularSparsePolynomial.zero(polynomial.variableCount, polynomial.modulus);
    }
    const groups = new Map();
    for (const term of polynomial.terms()) {
        const groupExponents = [...term.exponents];
        groupExponents[coefficientVariable] = 0n;
        const key = groupExponents.join(',');
        const coefficientExponents = new Array(polynomial.variableCount).fill(0n);
        coefficientExponents[coefficientVariable] = term.exponents[coefficientVariable];
        const group = groups.get(key) ?? [];
        group.push({
            coefficient: term.coefficient,
            exponents: coefficientExponents,
        });
        groups.set(key, group);
    }
    let content = ModularSparsePolynomial_1.ModularSparsePolynomial.zero(polynomial.variableCount, polynomial.modulus);
    for (const terms of groups.values()) {
        const coefficient = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, polynomial.modulus, terms);
        content = content.isZero()
            ? coefficient.monic('lex')
            : modularGcdUnivariate(content, coefficient, coefficientVariable);
        if (content.isConstant()) {
            return ModularSparsePolynomial_1.ModularSparsePolynomial.constant(polynomial.variableCount, polynomial.modulus, 1n);
        }
    }
    return content.monic('lex');
}
/** Removes polynomial content in one selected coefficient variable. */
function modularPrimitivePartInVariable(polynomial, coefficientVariable) {
    if (polynomial.isZero()) {
        return {
            content: ModularSparsePolynomial_1.ModularSparsePolynomial.zero(polynomial.variableCount, polynomial.modulus),
            primitive: polynomial,
        };
    }
    const content = modularContentInVariable(polynomial, coefficientVariable);
    const primitive = polynomial.divideExact(content);
    if (primitive === null) {
        throw new RangeError((0, errors_1.message)('polyModularContentExactDivision'));
    }
    return { content, primitive };
}
/**
 * Returns the coefficient polynomial at the lexicographically greatest monomial
 * in polynomialVariables. The result depends only on coefficientVariable.
 */
function modularLeadingCoefficientInVariables(polynomial, polynomialVariables, coefficientVariable) {
    polynomial.degree(coefficientVariable);
    for (const variableIndex of polynomialVariables) {
        polynomial.degree(variableIndex);
    }
    const terms = polynomial.terms();
    if (terms.length === 0) {
        return null;
    }
    let leading = terms[0];
    for (let termIndex = 1; termIndex < terms.length; termIndex++) {
        const term = terms[termIndex];
        for (const variableIndex of polynomialVariables) {
            if (term.exponents[variableIndex] === leading.exponents[variableIndex]) {
                continue;
            }
            if (term.exponents[variableIndex] > leading.exponents[variableIndex]) {
                leading = term;
            }
            break;
        }
    }
    const coefficientTerms = [];
    for (const term of terms) {
        let matches = true;
        for (const variableIndex of polynomialVariables) {
            if (term.exponents[variableIndex] !== leading.exponents[variableIndex]) {
                matches = false;
                break;
            }
        }
        if (!matches) {
            continue;
        }
        const exponents = new Array(polynomial.variableCount).fill(0n);
        exponents[coefficientVariable] = term.exponents[coefficientVariable];
        coefficientTerms.push({
            coefficient: term.coefficient,
            exponents,
        });
    }
    return new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, polynomial.modulus, coefficientTerms);
}
/**
 * Returns the monomial support over selected variables.
 *
 * Keys are used only for set membership and are never decoded back into exponents.
 */
function monomialSupport(polynomial, variables) {
    const support = new Set();
    for (const term of polynomial.terms()) {
        support.add(variables.map(variableIndex => term.exponents[variableIndex]).join(','));
    }
    return support;
}
/**
 * Computes the monic GCD of two univariate polynomials over the same modular field.
 *
 * The polynomial ring may retain additional variable coordinates, but neither input
 * may depend on variables other than variableIndex.
 */
function modularGcdUnivariate(left, right, variableIndex) {
    if (left.variableCount !== right.variableCount || left.modulus !== right.modulus) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular polynomial GCD', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
    }
    left.degree(variableIndex);
    right.degree(variableIndex);
    for (const activeVariable of left.variables()) {
        if (activeVariable !== variableIndex) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular polynomial GCD inputs', requirement: (0, errors_1.message)('polyReqUnivariatePolynomial') }));
        }
    }
    for (const activeVariable of right.variables()) {
        if (activeVariable !== variableIndex) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular polynomial GCD inputs', requirement: (0, errors_1.message)('polyReqUnivariatePolynomial') }));
        }
    }
    let a = new ModularSparsePolynomial_1.ModularSparsePolynomial(left.variableCount, left.modulus, left.terms());
    let b = new ModularSparsePolynomial_1.ModularSparsePolynomial(right.variableCount, right.modulus, right.terms());
    while (!b.isZero()) {
        const division = a.divideWithRemainder(b, 'lex');
        a = b;
        b = division.remainder;
    }
    return a.isZero() ? a : a.monic('lex');
}
/**
 * Computes a modular GCD recursively by specializing the last selected variable.
 *
 * The caller supplies the variable order used by the recursive polynomial view.
 * Ring coordinates not listed in variables must be dormant.
 */
function modularGcdRecursive(left, right, variables) {
    if (variables.length === 1) {
        return modularGcdUnivariate(left, right, variables[0]);
    }
    if (left.isZero()) {
        return right.isZero() ? right : right.monic('lex');
    }
    if (right.isZero()) {
        return left.monic('lex');
    }
    const interpolationVariable = variables[variables.length - 1];
    const polynomialVariables = variables.slice(0, -1);
    const leftParts = modularPrimitivePartInVariable(left, interpolationVariable);
    const rightParts = modularPrimitivePartInVariable(right, interpolationVariable);
    const contentGcd = modularGcdUnivariate(leftParts.content, rightParts.content, interpolationVariable);
    const leftLeading = modularLeadingCoefficientInVariables(leftParts.primitive, polynomialVariables, interpolationVariable);
    const rightLeading = modularLeadingCoefficientInVariables(rightParts.primitive, polynomialVariables, interpolationVariable);
    if (leftLeading === null || rightLeading === null) {
        return null;
    }
    const scaling = modularGcdUnivariate(leftLeading, rightLeading, interpolationVariable);
    const leftInterpolationDegree = leftParts.primitive.degree(interpolationVariable) ?? 0n;
    const rightInterpolationDegree = rightParts.primitive.degree(interpolationVariable) ?? 0n;
    const maxGcdDegree = leftInterpolationDegree < rightInterpolationDegree
        ? leftInterpolationDegree
        : rightInterpolationDegree;
    const leftLeadingDegree = leftLeading.degree(interpolationVariable) ?? 0n;
    const rightLeadingDegree = rightLeading.degree(interpolationVariable) ?? 0n;
    const scalingDegree = scaling.degree(interpolationVariable) ?? 0n;
    const requiredSamples = maxGcdDegree + 2n;
    const badPointBound = leftLeadingDegree + rightLeadingDegree + scalingDegree;
    const availablePoints = left.modulus - 1n;
    const requestedPoints = requiredSamples * MODULAR_GCD_RECONSTRUCTION_ATTEMPTS + badPointBound;
    const scanLimit = availablePoints < requestedPoints ? availablePoints : requestedPoints;
    let expectedDegrees = null;
    let expectedSupport = null;
    let acceptedSamples = 0n;
    let reconstructionAttempts = 0n;
    let samples = [];
    for (let value = 1n; value <= scanLimit; value++) {
        const leftLeadingValue = leftLeading.evaluateVariable(interpolationVariable, value);
        const rightLeadingValue = rightLeading.evaluateVariable(interpolationVariable, value);
        const scalingValue = scaling.evaluateVariable(interpolationVariable, value).constantTerm();
        if (leftLeadingValue.constantTerm() === 0n ||
            rightLeadingValue.constantTerm() === 0n ||
            scalingValue === 0n) {
            continue;
        }
        const evaluatedLeft = leftParts.primitive.evaluateVariable(interpolationVariable, value);
        const evaluatedRight = rightParts.primitive.evaluateVariable(interpolationVariable, value);
        if (evaluatedLeft.isZero() || evaluatedRight.isZero()) {
            continue;
        }
        const image = modularGcdRecursive(evaluatedLeft, evaluatedRight, polynomialVariables);
        if (image === null) {
            continue;
        }
        if (image.isConstant() && !image.isZero()) {
            return contentGcd;
        }
        const imageDegrees = polynomialVariables.map(variableIndex => image.degree(variableIndex) ?? 0n);
        if (expectedDegrees === null) {
            expectedDegrees = imageDegrees;
        }
        else {
            let degreeComparison = 0;
            for (let index = 0; index < imageDegrees.length; index++) {
                if (imageDegrees[index] < expectedDegrees[index]) {
                    degreeComparison = -1;
                    break;
                }
                if (imageDegrees[index] > expectedDegrees[index]) {
                    degreeComparison = 1;
                    break;
                }
            }
            if (degreeComparison < 0) {
                expectedDegrees = imageDegrees;
                expectedSupport = null;
                samples = [];
                acceptedSamples = 0n;
                reconstructionAttempts = 0n;
            }
            else if (degreeComparison > 0) {
                continue;
            }
        }
        const imageSupport = monomialSupport(image, polynomialVariables);
        if (expectedSupport === null) {
            expectedSupport = imageSupport;
        }
        else {
            const imageContainsExpected = [...expectedSupport].every(key => imageSupport.has(key));
            const currentSupport = expectedSupport;
            const expectedContainsImage = [...imageSupport].every(key => currentSupport.has(key));
            if (imageContainsExpected && !expectedContainsImage) {
                expectedSupport = imageSupport;
                samples = [];
                acceptedSamples = 0n;
                reconstructionAttempts = 0n;
            }
            else if (!imageContainsExpected) {
                continue;
            }
        }
        samples.push({
            value,
            polynomial: image.scale(scalingValue),
        });
        acceptedSamples++;
        if (acceptedSamples < requiredSamples) {
            continue;
        }
        const interpolated = ModularSparsePolynomial_1.ModularSparsePolynomial.interpolateVariable(interpolationVariable, samples);
        const reconstructed = modularPrimitivePartInVariable(interpolated, interpolationVariable).primitive;
        const candidate = reconstructed.multiply(contentGcd).monic('lex');
        if (left.divideExact(candidate) !== null && right.divideExact(candidate) !== null) {
            return candidate;
        }
        reconstructionAttempts++;
        if (reconstructionAttempts >= MODULAR_GCD_RECONSTRUCTION_ATTEMPTS) {
            return null;
        }
        samples = [];
        acceptedSamples = 0n;
    }
    return null;
}
/**
 * Computes a multivariate GCD over a prime field using recursive
 * evaluation and interpolation.
 *
 * variables must follow polynomial-ring order and contain every active variable
 * in both inputs exactly once. Dormant ring coordinates may be omitted.
 */
function modularGcdFiniteField(left, right, variables) {
    if (left.variableCount !== right.variableCount || left.modulus !== right.modulus) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular polynomial GCD', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
    }
    if (variables.length === 0) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular polynomial GCD', requirement: (0, errors_1.message)('polyReqAtLeastOneVariable') }));
    }
    if (!(0, primeFactor_1.isPrimeBig)(left.modulus)) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate modular GCD', requirement: (0, errors_1.message)('polyReqPrimeModulus') }));
    }
    const selected = new Set();
    let previousVariable = -1;
    for (const variableIndex of variables) {
        left.degree(variableIndex);
        right.degree(variableIndex);
        if (selected.has(variableIndex)) {
            throw new RangeError((0, errors_1.message)('polyVariablesUnique', { operation: 'Modular polynomial GCD' }));
        }
        if (variableIndex <= previousVariable) {
            throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Modular polynomial GCD' }));
        }
        selected.add(variableIndex);
        previousVariable = variableIndex;
    }
    for (const activeVariable of [...left.variables(), ...right.variables()]) {
        if (!selected.has(activeVariable)) {
            throw new RangeError((0, errors_1.message)('polyInputsSelectedOnly', { operation: 'Modular polynomial GCD' }));
        }
    }
    return modularGcdRecursive(left, right, variables);
}
/**
 * Computes a bivariate GCD over a prime field by evaluation and interpolation.
 */
function modularGcdBivariate(left, right, mainVariable, interpolationVariable) {
    if (mainVariable >= interpolationVariable) {
        throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Bivariate modular GCD' }));
    }
    return modularGcdFiniteField(left, right, [mainVariable, interpolationVariable]);
}
/**
 * Computes a trivariate GCD over a prime field by recursively specializing the
 * last selected variable and reconstructing from bivariate GCD images.
 *
 * The selected variables must follow their polynomial-ring order. Other ring
 * coordinates may remain dormant.
 */
function modularGcdTrivariate(left, right, mainVariable, secondaryVariable, interpolationVariable) {
    if (mainVariable >= secondaryVariable ||
        secondaryVariable >= interpolationVariable) {
        throw new RangeError((0, errors_1.message)('polyVariablesRingOrder', { operation: 'Trivariate modular GCD' }));
    }
    return modularGcdFiniteField(left, right, [
        mainVariable,
        secondaryVariable,
        interpolationVariable,
    ]);
}

};
modules["core/functions/bigint/primeFactor"]=function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isPrimeBig = isPrimeBig;
exports.nextPrimeBig = nextPrimeBig;
exports.trialDivisionBig = trialDivisionBig;
exports.pollardRhoBig = pollardRhoBig;
exports.primeFactorsBig = primeFactorsBig;
exports.primeFactorCounts = primeFactorCounts;
exports.mergeFactors = mergeFactors;
exports.getFactors = getFactors;
const bigint_1 = require("./bigint");
/**
 * Calculates and array of prime factors
 *
 * @see https://stackoverflow.com/questions/39899072/how-can-i-find-the-prime-factors-of-an-integer-in-javascript by @TheChiporpoise
 * @param num
 * @returns
 */
// export function primeFactors(num: bigint, toNumbers: true): { sign: number, factors: number[] };
// export function primeFactors(num: bigint, toNumbers: false | void): { sign: number, factors: bigint[] };
// export function primeFactors(num: bigint, toNumbers?: boolean | void) {
// 	const factors: bigint[] = [];
// 	const sgn = sign(num);
// 	num = abs(num)
// 	/* since 2 is the only even prime, it's easier to factor it out
// 	 * separately from the odd factor loop (for loop doesn't need to
// 	 * check whether or not to add 1 or 2 to f).
// 	 * The condition is essentially checking if the number is even
// 	 * (bitwise "&" operator compares the bits of 2 numbers in binary
// 	 * and outputs a binary number with 1's where their digits are the
// 	 * same and 0's where they differ. In this case it only checks if
// 	 * the final digit for num in binary is 1, which would mean the
// 	 * number is odd, in which case the output would be 1, which is
// 	 * interpreted as true, otherwise the output will be 0, which is
// 	 * interpreted as false. "!" returns the opposite boolean, so this
// 	 * means that '!(num & 1)' is true when the num is not odd)
// 	 */
// 	while (!(num & 1n)) {
// 		factors.push(2n);
// 		num /= 2n;
// 	}
// 	// 'f*f <= num' is faster than 'f <= Math.sqrt(num)'
// 	for (let f = 3n; f * f <= num; f += 2n) {
// 		while (!(num % f)) { // remainder of 'num / f' isn't 0
// 			factors.push(f);
// 			num /= f;
// 		}
// 	}
// 	/* if the number is already prime, then this adds it to factors so
// 	 * an empty array isn't returned
// 	 */
// 	if (num !== 1n) {
// 		factors.push(num);
// 	}
// 	return {
// 		sign: sgn,
// 		factors: toNumbers ? factors.map((n) => Number(n)) : factors
// 	}
// }
// Miller-Rabin primality test
function isPrimeBig(num) {
    if (num < 2n) {
        return false;
    }
    if (num === 2n || num === 3n) {
        return true;
    }
    if (num % 2n === 0n) {
        return false;
    }
    let r = 0n;
    let d = num - 1n;
    while (d % 2n === 0n) {
        r++;
        d /= 2n;
    }
    // These witnesses are deterministic for every unsigned 64-bit integer. For larger
    // BigInts they remain a stable probable-prime test rather than a randomized result.
    const witnesses = num < 18446744073709551616n
        ? [2n, 325n, 9375n, 28178n, 450775n, 9780504n, 1795265022n]
        : [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
    for (const witness of witnesses) {
        const a = witness % num;
        if (a < 2n) {
            continue;
        }
        let x = (0, bigint_1.powMod)(a, d, num);
        if (x === 1n || x === num - 1n) {
            continue;
        }
        let proceed = false;
        for (let j = 0n; j < r - 1n; j++) {
            x = (x * x) % num;
            if (x === num - 1n) {
                proceed = true;
                break;
            }
        }
        if (!proceed) {
            return false;
        }
    }
    return true;
}
/** Returns the least prime greater than or equal to start. */
function nextPrimeBig(start) {
    let candidate = start <= 2n ? 2n : start;
    if (candidate > 2n && candidate % 2n === 0n) {
        candidate++;
    }
    while (!isPrimeBig(candidate)) {
        candidate += candidate === 2n ? 1n : 2n;
    }
    return candidate;
}
/**
 * Fast prime factorization using trial division with optimizations
 */
function trialDivisionBig(n) {
    if (n <= 1n) {
        return [];
    }
    const factors = [];
    // Handle factor of 2
    while (n % 2n === 0n) {
        factors.push(2n);
        n /= 2n;
    }
    // Handle odd factors from 3 onwards
    let i = 3n;
    const sqrt = (num) => {
        if (num < 2n) {
            return num;
        }
        let x = num;
        let y = (x + 1n) / 2n;
        while (y < x) {
            x = y;
            y = (x + num / x) / 2n;
        }
        return x;
    };
    const limit = sqrt(n);
    while (i <= limit) {
        while (n % i === 0n) {
            factors.push(i);
            n /= i;
        }
        i += 2n; // Skip even numbers
    }
    // If n is still greater than 1, it's a prime factor
    if (n > 1n) {
        factors.push(n);
    }
    return factors;
}
/**
 * Pollard's rho algorithm for faster factorization of large numbers
 */
function pollardRhoBig(n) {
    if (n % 2n === 0n) {
        return 2n;
    }
    let x = 2n;
    let y = 2n;
    let d = 1n;
    const f = (val) => (val * val + 1n) % n;
    const gcd = (a, b) => {
        while (b !== 0n) {
            [a, b] = [b, a % b];
        }
        return a;
    };
    while (d === 1n) {
        x = f(x);
        y = f(f(y));
        d = gcd(x > y ? x - y : y - x, n);
    }
    return d === n ? 0n : d;
}
function primeFactorsBig(n, toNumbers) {
    const sgn = (0, bigint_1.sign)(n);
    n = (0, bigint_1.abs)(n);
    const factors = [];
    if (n === 1n || n === 2n || isPrimeBig(n)) {
        factors.push(n);
    }
    else {
        const factorize = (num) => {
            if (num === 1n) {
                return;
            }
            if (isPrimeBig(num)) {
                factors.push(num);
                return;
            }
            const divisor = pollardRhoBig(num);
            if (divisor === 0n) {
                // Fallback to trial division
                const simple = trialDivisionBig(num);
                factors.push(...simple);
                return;
            }
            factorize(divisor);
            factorize(num / divisor);
        };
        factorize(n);
    }
    factors.sort((a, b) => (a < b ? -1 : 1));
    return {
        sign: sgn,
        factors: toNumbers ? factors.map(x => Number(x)) : factors,
    };
}
/**
 * Places the prime factors in an object and counts their occurrences
 *
 * @param n
 */
function primeFactorCounts(n) {
    const counts = {};
    const factors = primeFactorsBig(n).factors;
    for (const factor of factors) {
        const f = String(factor);
        if (!(f in counts)) {
            counts[f] = 1n;
        }
        else {
            counts[f]++;
        }
    }
    return counts;
}
/**
 * Merges two factor counts into one object
 *
 * @param a
 * @param b
 */
function mergeFactors(a, b) {
    // Make a copy of a
    const retval = Object.assign(a, {});
    // Merge their counts
    for (const factor in b) {
        retval[factor] = factor in retval ? retval[factor] + b[factor] : b[factor];
    }
    return retval;
}
/**
 * Gets prime factors that can be used to create a number. The number one is
 * included since 1 * n is a possible combination.
 *
 * @param n The number for which the factors are being retrieved
 * @returns
 */
function getFactors(n) {
    const { sign: sgn, factors } = primeFactorsBig(BigInt(n), true);
    // Add the one since 1 * p is a combination
    factors.unshift(1);
    if (sgn === -1) {
        const l = factors.length;
        for (let i = 0; i < l; i++) {
            factors.push(-factors[i]);
        }
    }
    return factors;
}

};
modules["algebra/polynomial/ModularSparsePolynomialFactor"]=function(require,module,exports){
"use strict";
/**
 * @module modularSparsePolynomialFactor
 *
 * Finite-field factorization and Hensel lifting for sparse polynomials.
 *
 * External references consulted for Berlekamp factorization:
 * - SymPy finite-field factorization:
 *   https://github.com/sympy/sympy/blob/120ee85f346f6292b763cc195afda4f907260d4a/sympy/polys/galoistools.py
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.modularExtendedGcdUnivariate = modularExtendedGcdUnivariate;
exports.henselLiftMultivariateFactors = henselLiftMultivariateFactors;
exports.henselLiftUnivariateFactors = henselLiftUnivariateFactors;
exports.factorSquareFreeUnivariateFiniteField = factorSquareFreeUnivariateFiniteField;
const errors_1 = require("../../core/errors");
const bigint_1 = require("../../core/functions/bigint/bigint");
const primeFactor_1 = require("../../core/functions/bigint/primeFactor");
const modularGcd_1 = require("./modularGcd");
const ModularSparsePolynomial_1 = require("./ModularSparsePolynomial");
function assertSquareFreeUnivariate(polynomial, variableIndex) {
    if (!(0, primeFactor_1.isPrimeBig)(polynomial.modulus)) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular factorization', requirement: (0, errors_1.message)('polyReqPrimeModulus') }));
    }
    const degree = polynomial.degree(variableIndex);
    for (const activeVariable of polynomial.variables()) {
        if (activeVariable !== variableIndex) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular factorization', requirement: (0, errors_1.message)('polyReqUnivariatePolynomial') }));
        }
    }
    if (degree === null || degree === 0n) {
        return 0;
    }
    if (degree > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new RangeError((0, errors_1.message)('polyDegreeMatrixLimit'));
    }
    const derivativeTerms = [];
    for (const term of polynomial.terms()) {
        const exponent = term.exponents[variableIndex];
        if (exponent === 0n) {
            continue;
        }
        const exponents = [...term.exponents];
        exponents[variableIndex] = exponent - 1n;
        derivativeTerms.push({
            coefficient: (0, bigint_1.mod)(term.coefficient * exponent, polynomial.modulus),
            exponents,
        });
    }
    const derivative = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, polynomial.modulus, derivativeTerms);
    const gcd = (0, modularGcd_1.modularGcdUnivariate)(polynomial, derivative, variableIndex);
    if (!gcd.isConstant()) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Berlekamp factorization', requirement: (0, errors_1.message)('polyReqSquareFreeInput') }));
    }
    return Number(degree);
}
function polynomialRemainder(polynomial, modulusPolynomial) {
    return polynomial.divideWithRemainder(modulusPolynomial).remainder;
}
function polynomialPowMod(base, exponent, modulusPolynomial) {
    let result = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(base.variableCount, base.modulus, 1n);
    let factor = polynomialRemainder(base, modulusPolynomial);
    let power = exponent;
    while (power > 0n) {
        if ((power & 1n) === 1n) {
            result = polynomialRemainder(result.multiply(factor), modulusPolynomial);
        }
        power >>= 1n;
        if (power > 0n) {
            factor = polynomialRemainder(factor.multiply(factor), modulusPolynomial);
        }
    }
    return result;
}
function nullspaceModulo(matrix, modulus) {
    const rowCount = matrix.length;
    const columnCount = rowCount === 0 ? 0 : matrix[0].length;
    const rows = matrix.map(row => row.map(value => (0, bigint_1.mod)(value, modulus)));
    const pivotColumns = [];
    let pivotRow = 0;
    for (let column = 0; column < columnCount && pivotRow < rowCount; column++) {
        let selectedRow = pivotRow;
        while (selectedRow < rowCount && rows[selectedRow][column] === 0n) {
            selectedRow++;
        }
        if (selectedRow === rowCount) {
            continue;
        }
        if (selectedRow !== pivotRow) {
            const temp = rows[pivotRow];
            rows[pivotRow] = rows[selectedRow];
            rows[selectedRow] = temp;
        }
        const inverse = (0, bigint_1.invMod)(rows[pivotRow][column], modulus);
        if (inverse < 0n) {
            throw new RangeError((0, errors_1.message)('polyBerlekampNoninvertiblePivot'));
        }
        for (let c = column; c < columnCount; c++) {
            rows[pivotRow][c] = (0, bigint_1.mod)(rows[pivotRow][c] * inverse, modulus);
        }
        for (let row = 0; row < rowCount; row++) {
            if (row === pivotRow || rows[row][column] === 0n) {
                continue;
            }
            const scale = rows[row][column];
            for (let c = column; c < columnCount; c++) {
                rows[row][c] = (0, bigint_1.mod)(rows[row][c] - scale * rows[pivotRow][c], modulus);
            }
        }
        pivotColumns.push(column);
        pivotRow++;
    }
    const pivotSet = new Set(pivotColumns);
    const basis = [];
    for (let freeColumn = 0; freeColumn < columnCount; freeColumn++) {
        if (pivotSet.has(freeColumn)) {
            continue;
        }
        const vector = new Array(columnCount).fill(0n);
        vector[freeColumn] = 1n;
        for (let row = 0; row < pivotColumns.length; row++) {
            const pivotColumn = pivotColumns[row];
            vector[pivotColumn] = (0, bigint_1.mod)(-rows[row][freeColumn], modulus);
        }
        basis.push(vector);
    }
    return basis;
}
function vectorPolynomial(vector, variableCount, modulus, variableIndex) {
    const terms = [];
    for (let exponent = 0; exponent < vector.length; exponent++) {
        const coefficient = vector[exponent];
        if (coefficient === 0n) {
            continue;
        }
        const exponents = new Array(variableCount).fill(0n);
        exponents[variableIndex] = BigInt(exponent);
        terms.push({ coefficient, exponents });
    }
    return new ModularSparsePolynomial_1.ModularSparsePolynomial(variableCount, modulus, terms);
}
function changeModulus(polynomial, modulus) {
    return new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, modulus, polynomial.terms());
}
function integerPolynomialModulo(polynomial, modulus) {
    return new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, modulus, polynomial.terms());
}
function modularExtendedGcdUnivariate(left, right, variableIndex) {
    if (left.variableCount !== right.variableCount || left.modulus !== right.modulus) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular extended GCD', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
    }
    left.degree(variableIndex);
    right.degree(variableIndex);
    let oldR = left;
    let r = right;
    let oldS = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(left.variableCount, left.modulus, 1n);
    let s = ModularSparsePolynomial_1.ModularSparsePolynomial.zero(left.variableCount, left.modulus);
    let oldT = ModularSparsePolynomial_1.ModularSparsePolynomial.zero(left.variableCount, left.modulus);
    let t = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(left.variableCount, left.modulus, 1n);
    while (!r.isZero()) {
        const division = oldR.divideWithRemainder(r);
        const nextR = division.remainder;
        const nextS = oldS.subtract(division.quotient.multiply(s));
        const nextT = oldT.subtract(division.quotient.multiply(t));
        oldR = r;
        r = nextR;
        oldS = s;
        s = nextS;
        oldT = t;
        t = nextT;
    }
    const leading = oldR.leadingTerm();
    if (leading === null) {
        throw new RangeError((0, errors_1.message)('polyExtendedGcdTwoZeroUndefined'));
    }
    const inverse = (0, bigint_1.invMod)(leading.coefficient, left.modulus);
    if (inverse < 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Modular extended GCD', requirement: (0, errors_1.message)('polyReqInvertibleLeadingCoefficient') }));
    }
    return {
        gcd: oldR.scale(inverse),
        leftCoefficient: oldS.scale(inverse),
        rightCoefficient: oldT.scale(inverse),
    };
}
function henselPairStep(target, state) {
    const currentModulus = state.g.modulus;
    const nextModulus = currentModulus * currentModulus;
    const f = changeModulus(target, nextModulus);
    const g = changeModulus(state.g, nextModulus);
    const h = changeModulus(state.h, nextModulus);
    const s = changeModulus(state.s, nextModulus);
    const t = changeModulus(state.t, nextModulus);
    const error = f.subtract(g.multiply(h));
    const firstDivision = s.multiply(error).divideWithRemainder(h);
    const u = t
        .multiply(error)
        .add(firstDivision.quotient.multiply(g));
    const liftedG = g.add(u);
    const liftedH = h.add(firstDivision.remainder);
    const bezoutError = s
        .multiply(liftedG)
        .add(t.multiply(liftedH))
        .subtract(ModularSparsePolynomial_1.ModularSparsePolynomial.constant(f.variableCount, nextModulus, 1n));
    const secondDivision = s.multiply(bezoutError).divideWithRemainder(liftedH);
    const correction = t
        .multiply(bezoutError)
        .add(secondDivision.quotient.multiply(liftedG));
    return {
        g: liftedG,
        h: liftedH,
        s: s.subtract(secondDivision.remainder),
        t: t.subtract(correction),
    };
}
function productModularFactors(factors, variableCount, modulus) {
    let result = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(variableCount, modulus, 1n);
    for (const factor of factors) {
        result = result.multiply(changeModulus(factor, modulus));
    }
    return result;
}
function modularDerivativeOrder(polynomial, variableIndex, order) {
    let result = polynomial;
    let remaining = order;
    while (remaining > 0n) {
        result = result.derivative(variableIndex);
        remaining--;
    }
    return result;
}
function factorialModulo(value, modulus) {
    let result = 1n;
    let factor = 2n;
    while (factor <= value) {
        result = (0, bigint_1.mod)(result * factor, modulus);
        factor++;
    }
    return result;
}
function buildMultivariateHenselSeeds(leadingCoefficients, evaluationValues, baseFactors, variables, tailIndex, prime) {
    const mainVariable = variables[0];
    const tailVariable = variables[tailIndex + 1];
    const seeds = [];
    for (let index = 0; index < baseFactors.length; index++) {
        const baseFactor = baseFactors[index];
        const degree = baseFactor.degree(mainVariable);
        if (degree === null) {
            return null;
        }
        const lowerTerms = baseFactor
            .terms()
            .filter(term => term.exponents[mainVariable] !== degree);
        const lower = new ModularSparsePolynomial_1.ModularSparsePolynomial(baseFactor.variableCount, prime, lowerTerms);
        let leading = leadingCoefficients[index];
        for (let laterIndex = tailIndex + 1; laterIndex < evaluationValues.length; laterIndex++) {
            leading = leading.evaluateVariable(variables[laterIndex + 1], evaluationValues[laterIndex]);
        }
        const modularLeading = new ModularSparsePolynomial_1.ModularSparsePolynomial(leading.variableCount, prime, leading.terms());
        const mainExponents = new Array(baseFactor.variableCount).fill(0n);
        mainExponents[mainVariable] = degree;
        const mainPower = ModularSparsePolynomial_1.ModularSparsePolynomial.monomial(baseFactor.variableCount, prime, 1n, mainExponents);
        const seed = modularLeading.multiply(mainPower).add(lower);
        const specializedSeed = seed.evaluateVariable(tailVariable, evaluationValues[tailIndex]);
        if (!specializedSeed.equals(baseFactor)) {
            return null;
        }
        seeds.push(seed);
    }
    return seeds;
}
function solveUnivariateDiophantine(baseFactors, target, mainVariable) {
    const corrections = [];
    for (let index = 0; index < baseFactors.length; index++) {
        const others = baseFactors.filter((_factor, otherIndex) => otherIndex !== index);
        const complement = productModularFactors(others, baseFactors[index].variableCount, baseFactors[index].modulus);
        const bezout = modularExtendedGcdUnivariate(complement, baseFactors[index], mainVariable);
        if (!bezout.gcd.isConstant()) {
            return null;
        }
        const correction = target
            .multiply(bezout.leftCoefficient)
            .divideWithRemainder(baseFactors[index])
            .remainder;
        corrections.push(correction);
    }
    let reconstructed = ModularSparsePolynomial_1.ModularSparsePolynomial.zero(target.variableCount, target.modulus);
    for (let index = 0; index < baseFactors.length; index++) {
        const others = baseFactors.filter((_factor, otherIndex) => otherIndex !== index);
        const complement = productModularFactors(others, target.variableCount, target.modulus);
        reconstructed = reconstructed.add(corrections[index].multiply(complement));
    }
    if (!reconstructed.equals(target)) {
        return null;
    }
    return corrections;
}
function solveMultivariateDiophantine(baseFactors, target, mainVariable, tailVariables, evaluationValues, maxTailDegree) {
    if (tailVariables.length !== evaluationValues.length) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Diophantine solving', requirement: (0, errors_1.message)('polyReqOneEvaluationPerTailVariable') }));
    }
    if (tailVariables.length === 0) {
        return solveUnivariateDiophantine(baseFactors, target, mainVariable);
    }
    const tailVariable = tailVariables[tailVariables.length - 1];
    const evaluationValue = evaluationValues[evaluationValues.length - 1];
    const earlierVariables = tailVariables.slice(0, -1);
    const earlierValues = evaluationValues.slice(0, -1);
    const evaluatedFactors = baseFactors.map(factor => factor.evaluateVariable(tailVariable, evaluationValue));
    const evaluatedTarget = target.evaluateVariable(tailVariable, evaluationValue);
    const initial = solveMultivariateDiophantine(evaluatedFactors, evaluatedTarget, mainVariable, earlierVariables, earlierValues, maxTailDegree);
    if (initial === null) {
        return null;
    }
    const complements = baseFactors.map((_factor, index) => productModularFactors(baseFactors.filter((_other, otherIndex) => otherIndex !== index), target.variableCount, target.modulus));
    let corrections = [...initial];
    const tail = ModularSparsePolynomial_1.ModularSparsePolynomial.variable(target.variableCount, target.modulus, tailVariable);
    const shift = tail.subtract(ModularSparsePolynomial_1.ModularSparsePolynomial.constant(target.variableCount, target.modulus, evaluationValue));
    function residualFor(current) {
        let residual = target;
        for (let index = 0; index < current.length; index++) {
            residual = residual.subtract(current[index].multiply(complements[index]));
        }
        return residual;
    }
    let residual = residualFor(corrections);
    for (let order = 1n; order <= maxTailDegree && !residual.isZero(); order++) {
        const derivative = modularDerivativeOrder(residual, tailVariable, order).evaluateVariable(tailVariable, evaluationValue);
        const factorialInverse = (0, bigint_1.invMod)(factorialModulo(order, target.modulus), target.modulus);
        if (factorialInverse < 0n) {
            return null;
        }
        const coefficientTarget = derivative.scale(factorialInverse);
        if (coefficientTarget.isZero()) {
            continue;
        }
        const next = solveMultivariateDiophantine(evaluatedFactors, coefficientTarget, mainVariable, earlierVariables, earlierValues, maxTailDegree);
        if (next === null) {
            return null;
        }
        const shiftPower = shift.pow(order);
        corrections = corrections.map((correction, index) => correction.add(next[index].multiply(shiftPower)));
        residual = residualFor(corrections);
    }
    return residual.isZero() ? corrections : null;
}
/**
 * Lifts specialized factors through one or more tail variables modulo a prime.
 *
 * Callers supply the fully specialized factors, the corresponding main-variable
 * leading coefficients, and one evaluation value for each tail variable. The factors
 * are lifted one tail variable at a time and verified against the target after every
 * stage.
 */
function henselLiftMultivariateFactors(polynomial, factors, leadingCoefficients, variables, evaluationValues, prime) {
    if (variables.length < 2) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Hensel lifting', requirement: (0, errors_1.message)('polyReqAtLeastTwoSelectedVariables') }));
    }
    if (evaluationValues.length !== variables.length - 1) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Hensel lifting', requirement: (0, errors_1.message)('polyReqOneEvaluationPerTailVariable') }));
    }
    if (factors.length === 0 || factors.length !== leadingCoefficients.length) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Hensel lifting', requirement: (0, errors_1.message)('polyReqMatchingFactorLeadingLists') }));
    }
    if (!(0, primeFactor_1.isPrimeBig)(prime)) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Hensel lifting', requirement: (0, errors_1.message)('polyReqPrimeModulus') }));
    }
    const mainVariable = variables[0];
    let maxTailDegree = 0n;
    for (const variableIndex of variables.slice(1)) {
        const degree = polynomial.degree(variableIndex) ?? 0n;
        if (degree > maxTailDegree) {
            maxTailDegree = degree;
        }
    }
    if (prime <= maxTailDegree) {
        throw new RangeError((0, errors_1.message)('polyHenselPrimeExceedsTailDegree'));
    }
    for (const factor of [...factors, ...leadingCoefficients]) {
        if (factor.variableCount !== polynomial.variableCount) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Multivariate Hensel lifting inputs', requirement: (0, errors_1.message)('polyReqSamePolynomialRing') }));
        }
    }
    const target = new ModularSparsePolynomial_1.ModularSparsePolynomial(polynomial.variableCount, prime, polynomial.terms());
    let lifted = factors.map(factor => new ModularSparsePolynomial_1.ModularSparsePolynomial(factor.variableCount, prime, factor.terms()));
    let fullySpecializedTarget = target;
    for (let index = 0; index < evaluationValues.length; index++) {
        fullySpecializedTarget = fullySpecializedTarget.evaluateVariable(variables[index + 1], evaluationValues[index]);
    }
    if (!productModularFactors(lifted, target.variableCount, prime).equals(fullySpecializedTarget)) {
        return null;
    }
    for (let index = 0; index < lifted.length; index++) {
        for (let otherIndex = index + 1; otherIndex < lifted.length; otherIndex++) {
            const gcd = modularExtendedGcdUnivariate(lifted[index], lifted[otherIndex], mainVariable).gcd;
            if (!gcd.isConstant()) {
                return null;
            }
        }
    }
    for (let tailIndex = 0; tailIndex < variables.length - 1; tailIndex++) {
        const tailVariable = variables[tailIndex + 1];
        let stageTarget = target;
        for (let laterIndex = tailIndex + 1; laterIndex < evaluationValues.length; laterIndex++) {
            stageTarget = stageTarget.evaluateVariable(variables[laterIndex + 1], evaluationValues[laterIndex]);
        }
        const specializedTarget = stageTarget.evaluateVariable(tailVariable, evaluationValues[tailIndex]);
        if (!productModularFactors(lifted, target.variableCount, prime).equals(specializedTarget)) {
            return null;
        }
        const seeds = buildMultivariateHenselSeeds(leadingCoefficients, evaluationValues, lifted, variables, tailIndex, prime);
        if (seeds === null) {
            return null;
        }
        lifted = [...seeds];
        const tail = ModularSparsePolynomial_1.ModularSparsePolynomial.variable(target.variableCount, prime, tailVariable);
        const shift = tail.subtract(ModularSparsePolynomial_1.ModularSparsePolynomial.constant(target.variableCount, prime, evaluationValues[tailIndex]));
        const tailDegree = stageTarget.degree(tailVariable) ?? 0n;
        const baseFactors = lifted.map(factor => factor.evaluateVariable(tailVariable, evaluationValues[tailIndex]));
        for (let order = 1n; order <= tailDegree; order++) {
            const residual = stageTarget.subtract(productModularFactors(lifted, target.variableCount, prime));
            if (residual.isZero()) {
                break;
            }
            const derivative = modularDerivativeOrder(residual, tailVariable, order).evaluateVariable(tailVariable, evaluationValues[tailIndex]);
            const factorialInverse = (0, bigint_1.invMod)(factorialModulo(order, prime), prime);
            if (factorialInverse < 0n) {
                return null;
            }
            const correctionTarget = derivative.scale(factorialInverse);
            if (correctionTarget.isZero()) {
                continue;
            }
            const corrections = solveMultivariateDiophantine(baseFactors, correctionTarget, mainVariable, variables.slice(1, tailIndex + 1), evaluationValues.slice(0, tailIndex), maxTailDegree);
            if (corrections === null) {
                return null;
            }
            const shiftPower = shift.pow(order);
            lifted = lifted.map((factor, index) => factor.add(corrections[index].multiply(shiftPower)));
        }
        if (!productModularFactors(lifted, target.variableCount, prime).equals(stageTarget)) {
            return null;
        }
    }
    if (!productModularFactors(lifted, target.variableCount, prime).equals(target)) {
        return null;
    }
    return lifted.map(factor => factor.toSymmetricIntegerPolynomial()
        .primitivePart()
        .normalizeLeadingSign('lex'));
}
function liftFactorGroup(target, prime, primeFactors, variableIndex, targetModulus) {
    if (primeFactors.length === 1) {
        const targetAtModulus = changeModulus(target, targetModulus);
        const leading = targetAtModulus.leadingTerm();
        if (leading === null) {
            throw new RangeError((0, errors_1.message)('polyHenselZeroFactorGroup'));
        }
        const inverse = (0, bigint_1.invMod)(leading.coefficient, targetModulus);
        if (inverse < 0n) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqInvertibleLeadingCoefficient') }));
        }
        return [targetAtModulus.scale(inverse)];
    }
    const split = Math.floor(primeFactors.length / 2);
    const leftPrimeFactors = primeFactors.slice(0, split);
    const rightPrimeFactors = primeFactors.slice(split);
    const targetPrime = changeModulus(target, prime);
    const targetLeading = targetPrime.leadingTerm();
    if (targetLeading === null) {
        throw new RangeError((0, errors_1.message)('polyHenselCannotLiftZero'));
    }
    const leftProduct = productModularFactors(leftPrimeFactors, target.variableCount, prime).scale(targetLeading.coefficient);
    const rightProduct = productModularFactors(rightPrimeFactors, target.variableCount, prime);
    const bezout = modularExtendedGcdUnivariate(leftProduct, rightProduct, variableIndex);
    if (!bezout.gcd.isConstant()) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqPairwiseCoprimeModularFactors') }));
    }
    let state = {
        g: leftProduct,
        h: rightProduct,
        s: bezout.leftCoefficient,
        t: bezout.rightCoefficient,
    };
    while (state.g.modulus < targetModulus) {
        state = henselPairStep(target, state);
    }
    if (state.g.modulus !== targetModulus) {
        throw new Error((0, errors_1.message)('polyHenselModulusExceeded'));
    }
    return [
        ...liftFactorGroup(state.g, prime, leftPrimeFactors, variableIndex, targetModulus),
        ...liftFactorGroup(state.h, prime, rightPrimeFactors, variableIndex, targetModulus),
    ];
}
/**
 * Lifts pairwise-coprime monic factors from F_p[x] to a larger prime-power modulus.
 *
 * The achieved modulus is obtained by repeated quadratic Hensel steps, so callers
 * should pass a prime power of the form p^(2^k). The lifted factors remain monic and
 * reconstruct the integer polynomial after multiplication by its leading coefficient.
 */
function henselLiftUnivariateFactors(polynomial, prime, factors, variableIndex, targetModulus) {
    if (!(0, primeFactor_1.isPrimeBig)(prime)) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqPrimeBaseModulus') }));
    }
    if (factors.length === 0) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqAtLeastOneModularFactor') }));
    }
    if (targetModulus < prime) {
        throw new RangeError((0, errors_1.message)('polyHenselTargetNotSmaller'));
    }
    polynomial.degree(variableIndex);
    for (const activeVariable of polynomial.variables()) {
        if (activeVariable !== variableIndex) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqUnivariateIntegerPolynomial') }));
        }
    }
    for (const factor of factors) {
        if (factor.variableCount !== polynomial.variableCount ||
            factor.modulus !== prime) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqFactorsSameRingAndPrime') }));
        }
        for (const activeVariable of factor.variables()) {
            if (activeVariable !== variableIndex) {
                throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqUnivariateFactors') }));
            }
        }
        if (!factor.equals(factor.monic())) {
            throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqMonicModularFactors') }));
        }
    }
    let expectedModulus = prime;
    while (expectedModulus < targetModulus) {
        expectedModulus *= expectedModulus;
    }
    if (expectedModulus !== targetModulus) {
        throw new RangeError((0, errors_1.message)('polyHenselTargetRepeatedSquaring'));
    }
    const polynomialPrime = integerPolynomialModulo(polynomial, prime);
    const leading = polynomialPrime.leadingTerm();
    if (leading === null) {
        throw new RangeError((0, errors_1.message)('polyCannotFactorZeroPolynomial', { operation: 'Hensel lifting' }));
    }
    if ((0, bigint_1.invMod)(leading.coefficient, prime) < 0n) {
        throw new RangeError((0, errors_1.message)('polyRequires', { operation: 'Hensel lifting', requirement: (0, errors_1.message)('polyReqIntegerLeadingCoefficientInvertible') }));
    }
    const modularProduct = productModularFactors(factors, polynomial.variableCount, prime).scale(leading.coefficient);
    if (!modularProduct.equals(polynomialPrime)) {
        throw new RangeError((0, errors_1.message)('polyHenselFactorsReconstructModuloBase'));
    }
    const target = integerPolynomialModulo(polynomial, targetModulus);
    const lifted = liftFactorGroup(target, prime, factors, variableIndex, targetModulus);
    const targetLeading = target.leadingTerm();
    if (targetLeading === null) {
        throw new Error((0, errors_1.message)('polyHenselLeadingLost'));
    }
    const reconstructed = productModularFactors(lifted, polynomial.variableCount, targetModulus).scale(targetLeading.coefficient);
    if (!reconstructed.equals(target)) {
        throw new Error((0, errors_1.message)('polyHenselReconstructionFailed'));
    }
    return lifted;
}
/**
 * Factors a square-free univariate polynomial over a prime field with Berlekamp's method.
 *
 * The result contains monic irreducible factors whose product is the monic input.
 * Constants return an empty list.
 */
function factorSquareFreeUnivariateFiniteField(polynomial, variableIndex) {
    const degree = assertSquareFreeUnivariate(polynomial, variableIndex);
    if (degree === 0) {
        return [];
    }
    const input = polynomial.monic();
    if (degree === 1) {
        return [input];
    }
    const x = ModularSparsePolynomial_1.ModularSparsePolynomial.variable(input.variableCount, input.modulus, variableIndex);
    const xToP = polynomialPowMod(x, input.modulus, input);
    const matrix = Array.from({ length: degree }, () => new Array(degree).fill(0n));
    let columnPolynomial = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(input.variableCount, input.modulus, 1n);
    for (let column = 0; column < degree; column++) {
        for (let row = 0; row < degree; row++) {
            const exponents = new Array(input.variableCount).fill(0n);
            exponents[variableIndex] = BigInt(row);
            matrix[row][column] = columnPolynomial.coefficient(exponents);
        }
        matrix[column][column] = (0, bigint_1.mod)(matrix[column][column] - 1n, input.modulus);
        columnPolynomial = polynomialRemainder(columnPolynomial.multiply(xToP), input);
    }
    const kernel = nullspaceModulo(matrix, input.modulus);
    if (kernel.length <= 1) {
        return [input];
    }
    let factors = [input];
    for (const vector of kernel) {
        if (factors.length >= kernel.length) {
            break;
        }
        const splitter = vectorPolynomial(vector, input.variableCount, input.modulus, variableIndex);
        if (splitter.isConstant()) {
            continue;
        }
        let residue = 0n;
        while (residue < input.modulus && factors.length < kernel.length) {
            const shifted = splitter.subtract(ModularSparsePolynomial_1.ModularSparsePolynomial.constant(input.variableCount, input.modulus, residue));
            const refined = [];
            for (const factor of factors) {
                const factorDegree = factor.degree(variableIndex);
                if (factorDegree === null || factorDegree <= 1n) {
                    refined.push(factor);
                    continue;
                }
                const divisor = (0, modularGcd_1.modularGcdUnivariate)(factor, shifted, variableIndex);
                if (divisor.isConstant() || divisor.equals(factor)) {
                    refined.push(factor);
                    continue;
                }
                const quotient = factor.divideExact(divisor);
                if (quotient === null) {
                    throw new Error((0, errors_1.message)('polyBerlekampNonExactSplit'));
                }
                refined.push(divisor.monic(), quotient.monic());
            }
            factors = refined;
            residue++;
        }
    }
    if (factors.length !== kernel.length) {
        throw new Error((0, errors_1.message)('polyBerlekampIncompleteSplit'));
    }
    let reconstructed = ModularSparsePolynomial_1.ModularSparsePolynomial.constant(input.variableCount, input.modulus, 1n);
    for (const factor of factors) {
        reconstructed = reconstructed.multiply(factor);
    }
    if (!reconstructed.equals(input)) {
        throw new Error((0, errors_1.message)('polyBerlekampReconstructionFailed'));
    }
    return factors.sort((left, right) => {
        const leftDegree = left.degree(variableIndex) ?? 0n;
        const rightDegree = right.degree(variableIndex) ?? 0n;
        if (leftDegree !== rightDegree) {
            return leftDegree < rightDegree ? -1 : 1;
        }
        const leftTerms = left.terms();
        const rightTerms = right.terms();
        const count = Math.min(leftTerms.length, rightTerms.length);
        for (let index = 0; index < count; index++) {
            const leftExponent = leftTerms[index].exponents[variableIndex];
            const rightExponent = rightTerms[index].exponents[variableIndex];
            if (leftExponent !== rightExponent) {
                return leftExponent < rightExponent ? -1 : 1;
            }
            if (leftTerms[index].coefficient !== rightTerms[index].coefficient) {
                return leftTerms[index].coefficient < rightTerms[index].coefficient ? -1 : 1;
            }
        }
        return leftTerms.length - rightTerms.length;
    });
}

};
// Geometry-CAS opportunistic Z[x1,...,xn] factorization.
// Evaluation over F_65521 rejects nearly all linear candidates before any exact
// division; every accepted factor is verified in Z[x1,...,xn].  Unlike probabilistic
// irreducibility tests, modular rejections here cannot remove an actual factor.
const QUICK_MOD=65521;
const QUICK_SAMPLES=[[2,3,5,7,11,13],[17,19,23,29,31,37],[41,43,47,53,59,61]];
function quickSample(probe, index) {
    // A geometry construction can accumulate hundreds of dormant ring variables.
    return QUICK_SAMPLES[probe][index%6]+2*Math.floor(index/6);
}
const quickMod=x=>(x%QUICK_MOD+QUICK_MOD)%QUICK_MOD;
function quickInverse(x) {
    let t=0, newT=1, r=QUICK_MOD, newR=quickMod(x);
    while(newR) {
        const q=Math.floor(r/newR);
        [t,newT]=[newT,t-q*newT];
        [r,newR]=[newR,r-q*newR];
    }
    return quickMod(t);
}
const QUICK_COEFF=[-5,-3,-2,-1,0,1,2,3,5];
function quickGcd(a,b) {a=Math.abs(a);b=Math.abs(b);while(b){const r=a%b;a=b;b=r;}return a;}
function quickLinearSplit(poly, variables) {
    casBudgetCheck();
    const n=poly.variableCount;
    // A fast search is optional: its cost model selects whether to try
    // bounded-coefficient *candidates*, never whether the polynomial is factorable.
    // Unselected shapes always continue to the unrestricted original Wang/EEZ.
    if (variables.length===0 || variables.length>6 ||
        (poly.termCount>550 && variables.length>3) ||
        (poly.totalDegree()??0n)>16n) return null;
    const terms=[];
    let homogeneous=true,firstDegree=-1;
    const degrees=Array(n).fill(0);
    for(const t of poly.termsByKey.values()) {
        const e=t.exponents.map(Number);
        if(e.some(v=>!Number.isSafeInteger(v)||v>16))return null;
        const d=e.reduce((a,b)=>a+b,0);
        if(firstDegree<0) firstDegree=d;
        else if(firstDegree!==d) homogeneous=false;
        for(let j of variables) degrees[j]=Math.max(degrees[j],e[j]);
        terms.push({c:Number((t.coefficient%BigInt(QUICK_MOD)+BigInt(QUICK_MOD))%BigInt(QUICK_MOD)),e});
    }
    const maxDegrees=degrees;
    // Evaluate all but one variable at three independently chosen points.
    // A root of a linear factor must annihilate all three specialized polynomials.
    const projected=Array.from({length:n},()=>[]);
    for(const pivot of variables) {
        for(let probe=0;probe<3;probe++) {
            const powers=Array.from({length:n},()=>[1]);
            for(const j of variables) if(j!==pivot) {
                for(let k=1;k<=maxDegrees[j];k++)
                    powers[j][k]=powers[j][k-1]*quickSample(probe,j)%QUICK_MOD;
            }
            const coefficients=Array(maxDegrees[pivot]+1).fill(0);
            for(const t of terms) {
                let val=t.c;
                for(const j of variables) if(j!==pivot)
                    val=val*powers[j][t.e[j]]%QUICK_MOD;
                coefficients[t.e[pivot]]=(coefficients[t.e[pivot]]+val)%QUICK_MOD;
            }
            projected[pivot][probe]=coefficients;
        }
    }
    const vector=Array(n).fill(0);
    const optionCount=variables.length<=4 ? 4 : 3;
    const constants=homogeneous?[0]:variables.length<=4?[-5,-3,-2,-1,0,1,2,3,5]:[-1,0,1];
    let visited=0;
    function candidate() {
        if((++visited & 2047)===0)casBudgetCheck();
        // Exclude constant multiples: they cannot be primitive factors.
        const pivot=variables.find(i=>vector[i]!==0);
        if(pivot===undefined||vector[pivot]<0)return null;
        for(const c0 of constants) {
            let g=Math.abs(c0);
            for(const j of variables)g=quickGcd(g,vector[j]);
            if(g!==1)continue;
            // x alone was already removed by monomial-content extraction.
            if(c0===0 && variables.filter(i=>vector[i]).length===1)continue;
            const inv=quickInverse(vector[pivot]);
            let reject=false;
            for(let probe=0;probe<3;probe++) {
                let numerator=c0;
                for(const j of variables)if(j!==pivot)numerator+=vector[j]*quickSample(probe,j);
                const root=quickMod(-numerator*inv);
                const coefficients=projected[pivot][probe];
                let value=0;
                for(let k=coefficients.length-1;k>=0;k--)value=(value*root+coefficients[k])%QUICK_MOD;
                if(value!==0) {reject=true;break;}
            }
            if(reject)continue;
            const factorTerms=[];
            for(const j of variables)if(vector[j]) {
                const e=Array(n).fill(0n);e[j]=1n;
                factorTerms.push({coefficient:BigInt(vector[j]),exponents:e});
            }
            if(c0)factorTerms.push({coefficient:BigInt(c0),exponents:Array(n).fill(0n)});
            const factor=new P(n,factorTerms);
            const quotient=poly.divideExact(factor);
            if(quotient!==null && !quotient.isConstant())return {factor,quotient};
            if(quotient!==null && quotient.isConstant())return {factor,quotient};
        }
        return null;
    }
    function combinations(index,support,first) {
        if(support===0)return candidate();
        if(index>=variables.length || variables.length-index<support) return null;
        const j=variables[index];
        for(const coeff of QUICK_COEFF) {
            if(coeff===0)continue;
            vector[j]=coeff;
            const found=combinations(index+1,support-1,false);
            if(found)return found;
        }
        vector[j]=0;
        return combinations(index+1,support,first);
    }
    // First try few-supported factors. Sparsity is common in construction-geometry identities.
    for(let support=1;support<=Math.min(optionCount,variables.length);support++) {
        const found=combinations(0,support,true);
        if(found)return found;
    }
    return null;
}
// Leading-term sparse square root. This is an exact ring computation, not a
// floating-point square test. The only abort condition is the CAS deadline.
function geomIntegerSqrt(n) {
    if (n<0n) return null;
    if (n<9007199254740991n) {
        const x=BigInt(Math.floor(Math.sqrt(Number(n))));
        return x*x===n ? x : null;
    }
    let x=1n<<BigInt((n.toString(2).length+1)>>1);
    while (true) {
        casBudgetCheck();
        const y=(x+n/x)>>1n;
        if (y>=x) return x*x===n ? x : null;
        x=y;
    }
}
function geomExactSquareRoot(poly) {
    if (poly.isZero() || poly.isConstant()) return null;
    const top=poly.leadingTerm('lex');
    if (!top || top.coefficient<=0n || top.exponents.some(e=>e%2n!==0n))return null;
    const c=geomIntegerSqrt(top.coefficient);
    if(c===null)return null;
    const exponents=top.exponents.map(e=>e/2n);
    let root=P.monomial(poly.variableCount,c,exponents);
    while(true) {
        casBudgetCheck();
        const residue=poly.subtract(root.multiply(root));
        if(residue.isZero())return root;
        const t=residue.leadingTerm('lex');
        if(!t || t.coefficient%(2n*c)!==0n ||
           t.exponents.some((e,i)=>e<exponents[i]))return null;
        const e=t.exponents.map((v,i)=>v-exponents[i]);
        if(P.compareMonomials(e,exponents,'lex')>=0)return null;
        root=root.add(P.monomial(poly.variableCount,t.coefficient/(2n*c),e));
    }
}
function quickFactorMultivariateInteger(polynomial,variables) {
    if(!variables||variables.length<2)return F.factorUnivariateInteger(polynomial,variables?.[0]??0);
    if(polynomial.isZero())return {content:0n,factors:[]};
    casBudgetCheck();
    let content=polynomial.content();
    let remaining=polynomial.primitivePart();
    if((remaining.leadingTerm('lex')?.coefficient??1n)<0n) {
        content=-content;remaining=remaining.negate();
    }
    const factors=[];
    function append(factor,multiplicity) {
        if(multiplicity===0n||factor.isConstant())return;
        const known=factors.find(f=>f.factor.equals(factor));
        if(known)known.multiplicity+=multiplicity;
        else factors.push({factor,multiplicity});
    }
    // Remove monomial content directly, without invoking multivariate GCD.
    const minExp=Array(polynomial.variableCount).fill(null);
    for(const term of remaining.termsByKey.values()) for(const j of variables)
        if(minExp[j]===null||term.exponents[j]<minExp[j])minExp[j]=term.exponents[j];
    const shift=minExp.map(e=>e??0n);
    if(shift.some(e=>e!==0n)) {
        remaining=remaining.divideByMonomial({coefficient:1n,exponents:shift});
        for(const j of variables)if(shift[j])append(P.variable(polynomial.variableCount,j),shift[j]);
    }
    // Branch selection: an exact square is much cheaper than Wang/EEZ on
    // determinant identities, even with many active variables. No size cap.
    let squareMultiplicity=1n;
    try {
        while (!remaining.isConstant() && remaining.totalDegree()%2n===0n) {
            casBudgetCheck();
            const root=geomExactSquareRoot(remaining);
            if (!root || root.equals(remaining))break;
            remaining=root;
            squareMultiplicity*=2n;
        }
    } catch(err) {
        if (CAS_DEADLINE===Infinity || Date.now()<CAS_DEADLINE)throw err;
    }
    // Sparse, low-support linear candidates: F_p images reject cheaply.
    // The candidate alphabet is a heuristic, not a bound on the factorizer.
    if(!remaining.isConstant() && Date.now()<CAS_DEADLINE) {
        try {
            while(true) {
                casBudgetCheck();
                const found=quickLinearSplit(remaining,remaining.variables());
                if(!found)break;
                append(found.factor,squareMultiplicity);
                remaining=found.quotient;
                if(remaining.isConstant())break;
            }
        }catch(err) {
            if(CAS_DEADLINE===Infinity || Date.now()<CAS_DEADLINE)throw err;
        }
    }
    if(!remaining.isConstant()) {
        try {
            casBudgetCheck();
            const used=remaining.variables();
            // Wang/EEZ on every remainder, regardless of degree/support/terms.
            const deeper=used.length===1?F.factorUnivariateInteger(remaining,used[0]):M.factorMultivariateInteger(remaining,used);
            content*=deeper.content**squareMultiplicity;
            for(const entry of deeper.factors)append(entry.factor,entry.multiplicity*squareMultiplicity);
        } catch(err) {
            if(CAS_DEADLINE===Infinity || Date.now()<CAS_DEADLINE)throw err;
            append(remaining,squareMultiplicity);
        }
    }
    return {content,factors};
}
// Certified early coprimality: specialize every shared variable into F_p[x].
// For an actual multivariate common factor involving x, a degree-preserving
// specialization of both inputs can never have univariate gcd one.
function quickUnivariateImage(poly, pivot, sample) {
    const d=Number(poly.degree(pivot));
    if(!Number.isSafeInteger(d)||d>64)return null; // projection branch only
    const result=Array(d+1).fill(0);
    const table=Array.from({length:poly.variableCount},()=>[1]);
    const terms=poly.termsByKey.values();
    for(const t of terms) {
        let coefficient=Number((t.coefficient%BigInt(QUICK_MOD)+BigInt(QUICK_MOD))%BigInt(QUICK_MOD));
        for(let j=0;j<poly.variableCount;j++)if(j!==pivot) {
            const exp=Number(t.exponents[j]);
            if(!Number.isSafeInteger(exp)||exp>64)return null;
            const row=table[j];
            while(row.length<=exp)row.push(row[row.length-1]*sample[j]%QUICK_MOD);
            coefficient=coefficient*row[exp]%QUICK_MOD;
        }
        const e=Number(t.exponents[pivot]);
        result[e]=(result[e]+coefficient)%QUICK_MOD;
    }
    return result[d]===0?null:result;
}
function quickUnivariateCoprime(a,b) {
    const trim=x=>{while(x.length&&x[x.length-1]===0)x.pop();return x;};
    a=trim(a.slice());b=trim(b.slice());
    while(b.length>1) {
        const invert=quickInverse(b[b.length-1]);
        let r=a.slice();
        while(r.length>=b.length) {
            const c=r[r.length-1]*invert%QUICK_MOD;
            const delta=r.length-b.length;
            for(let i=0;i<b.length;i++) r[delta+i]=quickMod(r[delta+i]-c*b[i]);
            trim(r);
        }
        a=b;b=r;
    }
    return b.length===1 || (b.length===0&&a.length===1);
}
function quickCoprimeCertificate(a,b) {
    if(a.isZero()||b.isZero()||a.isConstant()||b.isConstant())return false;
    // Projection can be expensive for dense inputs, but this is merely a
    // heuristic filter. The modular GCD remains available for any input size.
    if(a.termCount*b.termCount>300000)return false;
    const av=a.variables(),bv=new Set(b.variables());
    for(const pivot of av) {
        if(!bv.has(pivot))continue;
        let proved=false;
        for(let probe=0;probe<3;probe++) {
            const sample=Array.from({length:a.variableCount},(_,j)=>quickSample(probe,j));
            const ia=quickUnivariateImage(a,pivot,sample);
            if(ia===null)continue;
            const ib=quickUnivariateImage(b,pivot,sample);
            if(ib===null)continue;
            if(quickUnivariateCoprime(ia,ib)){proved=true;break;}
        }
        if(!proved)return false;
    }
    return true;
}
function quickBigGcd(a,b) {
    a=a<0n?-a:a;b=b<0n?-b:b;
    while(b!==0n){const r=a%b;a=b;b=r;}
    return a;
}
function quickPolynomialGcd(a,b,variables) {
    // Do not weaken Nerdamer's variable-order and ring validation contract.
    if(a.variableCount===b.variableCount && Array.isArray(variables) &&
       variables.length>0 && variables.every((v,i)=>Number.isInteger(v)&&v>=0&&v<a.variableCount&&(i===0||v>variables[i-1])) &&
       [...a.variables(),...b.variables()].every(v=>variables.includes(v))) {
        // Cheap exact divisibility wins particularly for repeated factors.
        // Valid for primitive parts even when their integer contents differ.
        const ac=a.content(),bc=b.content();
        if(!a.isZero()&&!b.isZero()&&!a.isConstant()&&!b.isConstant()){
            const x=a.primitivePart().normalizeLeadingSign('lex');
            const y=b.primitivePart().normalizeLeadingSign('lex');
            const [shorter,longer]=x.termCount<=y.termCount?[x,y]:[y,x];
            if(shorter.termCount<=longer.termCount &&
               (shorter.totalDegree()??0n)<=(longer.totalDegree()??0n) &&
               shorter.termCount*longer.termCount<25000) {
                casBudgetCheck();
                if(longer.divideExact(shorter)!==null)
                    return shorter.scale(quickBigGcd(ac,bc));
            }
        }
        // Certificate is one-way: failure is NOT a proof of common factors.
        if(quickCoprimeCertificate(a,b))
            return P.constant(a.variableCount,quickBigGcd(ac,bc));
    }
    return G.sparsePolynomialGcd(a,b,variables);
}
const P=load('core/classes/polynomial/SparsePolynomial').SparsePolynomial;
const F=load('algebra/polynomial/SparsePolynomialFactor');
const M=load('algebra/polynomial/SparsePolynomialMultivariateFactor');
const G=load('algebra/polynomial/SparsePolynomialGcd');
root.GeomNerdamerSparse2={Polynomial:P,factorUnivariateInteger:F.factorUnivariateInteger,
    factorMultivariateInteger:quickFactorMultivariateInteger,
    sparsePolynomialGcd:quickPolynomialGcd,
    setOperationDeadline(msEpoch){CAS_DEADLINE=msEpoch;},
    factorMultivariateOriginal:M.factorMultivariateInteger};
})(typeof self!=='undefined'?self:globalThis);
