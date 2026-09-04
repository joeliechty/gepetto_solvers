#pragma once

// Signed distance from a point to one ellipsoid's surface, in that ellipsoid's
// own frame. The single definition every ellipsoid factor measures with.

#include <gtsam/base/Matrix.h>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>

namespace gepetto_solvers {

// TWO METRICS, ONE SIGN CONVENTION. Both forms below return a SIGNED distance to
// the surface x^T M x = 1 (M = diag(a^-2, b^-2, c^-2)) -- negative inside,
// positive outside, zero on it -- and both carry an exact analytic gradient. They
// share a zero set exactly; they differ in what they report away from it.
//
//   EXACT (the default, Eberly [eberly2008distance]). The true orthogonal
//   distance ||u - y|| to the closest surface point y, found by one bracketed
//   Newton solve. Its gradient is the UNIT surface normal at y, so ||grad|| == 1
//   everywhere: the Jacobian rows this feeds are equally conditioned at every
//   eccentricity, which is what lets the AL solver resolve contact on a flat or
//   coin-like object at the same step size it uses on a sphere.
//
//   TAUBIN (taubin=true, the previous behaviour). The first-order algebraic
//   approximation (x^T M x - 1) / (2 ||M x||) [taubin1991estimation]. Cheap and
//   C-infinity, and correct to first order near the surface -- but its gradient
//   norm drifts from 1 with eccentricity, which is the ill-conditioning the exact
//   form exists to remove. Kept because it is what every result before this
//   change was produced with, and because a smooth-everywhere metric is
//   occasionally the safer one: the exact distance's gradient is only C^0 across
//   the interior medial axis, where the closest surface point jumps.
//
// Both are ~equally cheap in practice next to the Pose3 chain rule around them,
// so the choice is about conditioning, not speed.

namespace ellipsoid_detail {

// A component of |x| below this fraction of its own semi-axis is treated as
// exactly this fraction instead. Zero components are the degenerate case of the
// Eberly parameterization -- they remove the pole that brackets the root and
// make y_j = e_j^2 u_j / (t + e_j^2) a 0/0 -- and nudging them off zero is both
// cheaper and better conditioned than a separate branch per vanishing component.
//
// The value trades two errors against each other and is nowhere near either
// extreme. Down: the nudge displaces the query point by this fraction of a
// semi-axis, half a nanometre on a 5 cm axis, and the distance depends on a
// transverse displacement only to SECOND order, so it costs nothing measurable.
// Up: as x approaches the centre the root approaches the pole at -min_j e_j^2,
// and t + e_j^2 is then a difference of two numbers ~1e5 times larger than
// itself -- cancellation whose relative error is eps/this. At 1e-12 the exact
// centre of a 10 mm axis came out 18 um wrong; at 1e-8 that is 0.2 nm, and the
// nudge itself is still 0.5 nm. Both sit far below any length this solver cares
// about, which is the point: there is no value in between worth tuning.
constexpr double kMinAxisFraction = 1e-8;

// Below this the query point is ON the surface, where u - y vanishes and its
// direction is 0/0. The algebraic normal is exact there, so use it.
constexpr double kSurfaceEpsilon = 1e-12;

constexpr int    kMaxIterations  = 100;

// Relative convergence on t. The scale is min_j e_j^2 -- the SMALLEST squared
// semi-axis, because the root's left bracket is the pole at -min_j e_j^2 and
// what has to be resolved is t's distance from it, not t's own magnitude. On an
// eccentric shape those differ by the eccentricity squared: scaling by the
// largest axis instead stops the iteration ~1e4 too early on a 200:1 ellipsoid,
// and the error lands in y_j = e_j^2 u_j / (t + e_j^2) as a bad closest point.
//
// It is worth converging all the way -- dy/dt grows without bound as the query
// point approaches the interior medial axis, so slack here is amplified, and
// Newton doubles its digits per step, which makes the last iterations nearly
// free. The t_next == t test below is what actually terminates at the end.
constexpr double kRootTolerance  = 1e-16;

// Taubin's first-order distance and its exact gradient.
//   d     = f / (2 g),   f = x^T M x - 1,  g = ||M x||
//   dd/dx = (M x)^T / g - f / (2 g^3) (M M x)^T
inline double taubin_signed_distance(const gtsam::Vector3& m_diag,
                                     const gtsam::Vector3& x,
                                     gtsam::Matrix13* H)
{
    const gtsam::Vector3 Mx = m_diag.cwiseProduct(x);
    const double f = x.dot(Mx) - 1.0;
    double g = Mx.norm();
    if (g < 1e-9) g = 1e-9;   // the centre; the gradient has no direction there

    if (H) {
        const gtsam::Vector3 mMx = m_diag.cwiseProduct(Mx);
        *H = (Mx.transpose() / g) - (f / (2.0 * g * g * g)) * mMx.transpose();
    }
    return f / (2.0 * g);
}

// The unique root of F(t) = sum_j c_j / (t + e2_j)^2 - 1 inside [t_lo, t_hi],
// which the caller has already bracketed with F(t_lo) >= 0 >= F(t_hi).
//
// F is strictly decreasing wherever every t + e2_j > 0, so Newton converges
// quadratically from either end -- but F has a pole just left of the interior
// bracket, and a Newton step off the tangent there can overshoot straight past
// it. Every step is therefore clamped into a bracket that is itself narrowed
// from the sign of F, so the iteration degrades to bisection rather than
// diverging, and terminates for any input.
inline double eberly_root(const gtsam::Vector3& e2, const gtsam::Vector3& c,
                          double t_lo, double t_hi)
{
    const double t_scale = e2.minCoeff();
    double t = 0.5 * (t_lo + t_hi);
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        double F = -1.0, dF = 0.0;
        for (int j = 0; j < 3; ++j) {
            const double den = t + e2[j];
            const double r = c[j] / (den * den);
            F  += r;
            dF -= 2.0 * r / den;
        }

        if (F > 0.0) t_lo = t;
        else         t_hi = t;

        double t_next = (dF < 0.0) ? (t - F / dF)
                                   : 0.5 * (t_lo + t_hi);
        if (!(t_next > t_lo && t_next < t_hi))
            t_next = 0.5 * (t_lo + t_hi);

        // The second test is the one that actually fires at the end: once the
        // step is below an ulp of t, Newton stops moving and the first test may
        // never be met exactly.
        const bool converged =
            std::abs(t_next - t) <= kRootTolerance * (t_scale + std::abs(t))
            || t_next == t;
        t = t_next;
        if (converged) break;
    }
    return t;
}

// Eberly's exact orthogonal distance, signed by which side of the surface x is on.
//
// Fold x into the first octant as u = |x| (the ellipsoid is symmetric in every
// axis plane, so the closest point transforms with it), where the closest surface
// point is y_j = e_j^2 u_j / (t + e_j^2) for the unique root t of
//
//   F(t) = sum_j ( e_j u_j / (t + e_j^2) )^2 - 1 = 0
//
// on (-min_j e_j^2, inf). F falls from +inf to -1 across that interval, so the
// root is unique -- positive outside the ellipsoid, negative inside -- and ONE
// bracketed solve covers both sides. That matters here: these factors are
// penetration constraints, so the inside is not an edge case.
//
// The gradient is the unit outward normal at y, mapped back out of the octant:
//   dd/dx = diag(sgn(x)) * sign(f) * (u - y) / ||u - y||
// with sign(f) flipping the (inward-pointing) u - y for an interior point.
inline double exact_signed_distance(const gtsam::Vector3& semi_axes,
                                    const gtsam::Vector3& m_diag,
                                    const gtsam::Vector3& x,
                                    gtsam::Matrix13* H)
{
    gtsam::Vector3 u, sgn;
    for (int j = 0; j < 3; ++j) {
        u[j]   = std::max(std::abs(x[j]), kMinAxisFraction * semi_axes[j]);
        sgn[j] = (x[j] < 0.0) ? -1.0 : 1.0;
    }

    // Which side, and by the algebraic test rather than the solved distance:
    // it is the sign the Taubin form uses, so the two metrics agree on the
    // inside/outside question exactly and a switch of flag cannot flip it.
    const double f = u.dot(m_diag.cwiseProduct(u)) - 1.0;
    const double s = (f < 0.0) ? -1.0 : 1.0;

    const gtsam::Vector3 e2 = semi_axes.cwiseProduct(semi_axes);
    const gtsam::Vector3 eu = semi_axes.cwiseProduct(u);
    const gtsam::Vector3 c  = eu.cwiseProduct(eu);            // (e_j u_j)^2

    // Bracket the root, using the side we already know.
    //   outside: F(0) = f >= 0, and F(||e.u||) < sum c_j / ||e.u||^2 - 1 = 0.
    //   inside:  F(0) = f < 0, and at t = -e_m^2 + e_m u_m (e_m the SMALLEST
    //            semi-axis) that member's term alone is 1, so F >= 0. Interior
    //            points have u_m < e_m, so this lower bound stays below zero,
    //            and t + e_j^2 > 0 for every j -- y is finite throughout.
    double t_lo, t_hi;
    if (f >= 0.0) {
        t_lo = 0.0;
        t_hi = eu.norm();
    } else {
        int m = 0;
        if (semi_axes[1] < semi_axes[m]) m = 1;
        if (semi_axes[2] < semi_axes[m]) m = 2;
        t_lo = -e2[m] + semi_axes[m] * u[m];
        t_hi = 0.0;
    }

    const double t = eberly_root(e2, c, t_lo, t_hi);

    // u_j - y_j, written as u_j t / (t + e_j^2) rather than as the subtraction
    // it is algebraically equal to. Near the surface t -> 0, so u_j and y_j
    // agree to nearly every digit and differencing them throws away exactly the
    // ones that carry the answer -- and near the surface is where a contact
    // equality spends its whole solve. This form loses nothing: at t = 0 it is
    // exactly 0, which is the right answer rather than a rounding of it.
    gtsam::Vector3 diff;
    for (int j = 0; j < 3; ++j)
        diff[j] = u[j] * t / (t + e2[j]);
    const double dist = diff.norm();

    if (H) {
        if (dist > kSurfaceEpsilon) {
            *H = (s / dist) * sgn.cwiseProduct(diff).transpose();
        } else {
            // On the surface: u - y is zero, but the outward normal there is
            // grad(x^T M x) = 2 M x, normalized -- which is the same vector.
            gtsam::Vector3 n = m_diag.cwiseProduct(x);
            const double nn = n.norm();
            n = (nn > 1e-12) ? (n / nn).eval() : gtsam::Vector3(0.0, 0.0, 1.0);
            *H = n.transpose();
        }
    }
    return s * dist;
}

}  // namespace ellipsoid_detail

// One ellipsoid's signed distance field, in the ellipsoid's OWN frame: the
// caller transforms the query point in and chains the Jacobian back out.
//
// It owns the shape constants (M and M^-1) the factors used to each cache for
// themselves, so a factor holds one of these per member instead of parallel
// vectors of diagonals -- and there is exactly one place where the choice of
// metric is made.
class EllipsoidDistance {
public:
    // `taubin` selects the legacy algebraic approximation over the exact
    // orthogonal distance; see the TWO METRICS note above.
    EllipsoidDistance(const gtsam::Vector3& semi_axes, bool taubin)
        : semi_axes_(semi_axes), taubin_(taubin)
    {
        // A non-positive semi-axis is not a flat ellipsoid, it is a division by
        // zero that reaches the residual as a NaN several frames later.
        if (!(semi_axes.x() > 0.0 && semi_axes.y() > 0.0 && semi_axes.z() > 0.0))
            throw std::invalid_argument(
                "EllipsoidDistance: every semi-axis must be > 0 (got " +
                std::to_string(semi_axes.x()) + ", " +
                std::to_string(semi_axes.y()) + ", " +
                std::to_string(semi_axes.z()) + ")");

        m_diag_    = gtsam::Vector3(1.0 / (semi_axes.x() * semi_axes.x()),
                                    1.0 / (semi_axes.y() * semi_axes.y()),
                                    1.0 / (semi_axes.z() * semi_axes.z()));
        minv_diag_ = semi_axes.cwiseProduct(semi_axes);
    }

    // Signed distance from `x` (this ellipsoid's frame) to the surface: negative
    // inside, positive outside. *H, when given, is d(distance)/dx as a row.
    double signed_distance(const gtsam::Vector3& x,
                           gtsam::Matrix13* H = nullptr) const
    {
        return taubin_
            ? ellipsoid_detail::taubin_signed_distance(m_diag_, x, H)
            : ellipsoid_detail::exact_signed_distance(semi_axes_, m_diag_, x, H);
    }

    const gtsam::Vector3& semi_axes() const { return semi_axes_; }
    const gtsam::Vector3& m_diag()    const { return m_diag_; }     // diag(a^-2, ...)
    const gtsam::Vector3& minv_diag() const { return minv_diag_; }  // diag(a^2, ...)
    bool taubin() const { return taubin_; }

    // The outward unit surface normal the ALGEBRAIC form gives at x, i.e.
    // normalize(M x). Exact on the surface for either metric, which is where the
    // contact factors read it, and independent of the metric flag so the normal
    // rows of a residual do not move when the distance row's metric does.
    gtsam::Vector3 algebraic_normal(const gtsam::Vector3& x) const {
        gtsam::Vector3 n = m_diag_.cwiseProduct(x);
        const double nn = n.norm();
        if (nn > 1e-8) return n / nn;
        return gtsam::Vector3(0.0, 0.0, 1.0);
    }

private:
    gtsam::Vector3 semi_axes_;
    gtsam::Vector3 m_diag_;
    gtsam::Vector3 minv_diag_;
    bool taubin_;
};

}  // namespace gepetto_solvers
