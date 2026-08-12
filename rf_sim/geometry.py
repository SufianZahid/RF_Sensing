"""2D geometry utilities for line segment intersection and wall detection."""

from typing import List, Tuple
from rf_sim.environment import Wall

Point2D = Tuple[float, float]
Segment = Tuple[Point2D, Point2D]


def _orientation(p: Point2D, q: Point2D, r: Point2D) -> float:
    """Computes the cross product orientation of ordered triplet (p, q, r).

    Returns:
        float: 0 if collinear, > 0 if clockwise, < 0 if counter-clockwise.
    """
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def _on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """Given three collinear points p, q, r, checks if point q lies on segment pr."""
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def segment_intersects(seg1: Segment, seg2: Segment) -> bool:
    """Determines whether two 2D line segments seg1 and seg2 intersect.

    Args:
        seg1: Tuple of points ((x1, y1), (x2, y2)) representing the first segment.
        seg2: Tuple of points ((x3, y3), (x4, y4)) representing the second segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    p1, p2 = seg1
    q1, q2 = seg2

    # Find the 4 orientations needed for general and special cases
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # Floating point epsilon for orientation checks
    eps = 1e-9

    # General case
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    # Special collinear cases
    if abs(o1) <= eps and _on_segment(p1, q1, p2):
        return True
    if abs(o2) <= eps and _on_segment(p1, q2, p2):
        return True
    if abs(o3) <= eps and _on_segment(q1, p1, q2):
        return True
    if abs(o4) <= eps and _on_segment(q1, p2, q2):
        return True

    return False


def walls_between(p1: Point2D, p2: Point2D, walls: List[Wall]) -> List[Wall]:
    """Finds all walls that intersect the direct line segment between p1 and p2.

    Args:
        p1: Transmitter (x, y) coordinates.
        p2: Receiver (x, y) coordinates.
        walls: List of Wall instances in the environment.

    Returns:
        List[Wall]: List of Wall objects intersected by segment p1->p2.
    """
    ray_segment = (p1, p2)
    intersected: List[Wall] = []
    for wall in walls:
        wall_segment = (wall.start_point, wall.end_point)
        if segment_intersects(ray_segment, wall_segment):
            intersected.append(wall)
    return intersected
