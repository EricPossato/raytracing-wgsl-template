fn hit_sphere(center: vec3f, radius: f32, r: ray, record: ptr<function, hit_record>, max: f32) -> bool {
    var oc = r.origin - center;
    var a = dot(r.direction, r.direction);
    var b = dot(oc, r.direction);
    var c = dot(oc, oc) - radius * radius;
    var disc = b * b - a * c;

    if (disc <= 0.0) {
        return false;
    }
    var sqrt_disc = sqrt(disc);
    var t = (-b - sqrt_disc) / a;
    if (t < max && t > RAY_TMIN) {
        (*record).t = t;
        (*record).p = ray_at(r, t);
        (*record).normal = ((*record).p - center) / radius;
        (*record).hit_anything = true;
        return true;
    }
    return false;
}

fn hit_quad(r: ray, Q: vec4f, u: vec4f, v: vec4f, record: ptr<function, hit_record>, max: f32) -> bool {
    var n = cross(u.xyz, v.xyz);
    n = normalize(n);
    var den = dot(n, r.direction);
    if (abs(den) > RAY_TMIN) {
        var t = dot(Q.xyz - r.origin, n) / den;
        if (t > RAY_TMIN && t < max) {
            let p = ray_at(r, t);
            let d = p - Q.xyz;
            let uu_dot = dot(u.xyz, u.xyz);
            let uv_dot = dot(u.xyz, v.xyz);
            let vv_dot = dot(v.xyz, v.xyz);
            let du_dot = dot(d, u.xyz);
            let dv_dot = dot(d, v.xyz);

            let inv_den = 1.0 / (uu_dot * vv_dot - uv_dot * uv_dot);
            let u_coord = (vv_dot * du_dot - uv_dot * dv_dot) * inv_den;
            let v_coord = (uu_dot * dv_dot - uv_dot * du_dot) * inv_den;

            if (u_coord >= 0.0 && v_coord >= 0.0 && u_coord <= 1.0  && v_coord <= 1.0) {
                (*record).t = t;
                (*record).p = p;
                (*record).normal = n;
                (*record).hit_anything = true;
                return true;
            }
        }
    }
    return false;
}

fn hit_triangle(r: ray, v0: vec3f, v1: vec3f, v2: vec3f, record: ptr<function, hit_record>, max: f32)
{
  var v1v0 = v1 - v0;
  var v2v0 = v2 - v0;
  var rov0 = r.origin - v0;

  var n = cross(v1v0, v2v0);
  var q = cross(rov0, r.direction);

  var d = 1.0 / dot(r.direction, n);

  var u = d * dot(-q, v2v0);
  var v = d * dot(q, v1v0);
  var t = d * dot(-n, rov0);

  if (u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0)
  {
    record.hit_anything = false;
    return;
  }

  if (t < RAY_TMIN || t > max)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = normalize(n);
  record.hit_anything = true;
}

fn hit_box(r: ray, center: vec3f, rad: vec3f, rotation: vec3f, record: ptr<function, hit_record>, t_max: f32) -> bool
{
  var rot = rotation.xyz;//assuming rotation received is already in radians
  var quat = quaternion_from_euler(rot);

  var m = 1.0 / r.direction;
  var n = m * (r.origin - center);
  var k = abs(m) * rad;

  var t1 = -n - k;
  var t2 = -n + k;

  var tN = max(max(t1.x, t1.y), t1.z);
  var tF = min(min(t2.x, t2.y), t2.z);

  if (tN > tF || tF < 0.0)
  {
    record.hit_anything = false;
    return false;
  }

  var t: f32;
  if (tN > 0.0)
  {
    t = tN;
  }
  else{
    t = tF;
  }

  if (t < RAY_TMIN || t > t_max)
  {
    record.hit_anything = false;
    return false;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = -sign(r.direction) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  record.hit_anything = true;

  return true;
}