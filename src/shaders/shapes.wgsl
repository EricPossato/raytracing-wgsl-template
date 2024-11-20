fn hit_sphere(center: vec3f, radius: f32, r: ray, record: ptr<function, hit_record>, max: f32) -> bool {
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let b = dot(oc, r.direction);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - a * c;

    if (disc <= 0.0) {
        return false;
    }
    let sqrt_disc = sqrt(disc);
    let t = (-b - sqrt_disc) / a;
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
    let n = normalize(cross(u.xyz, v.xyz));
    let denom = dot(n, r.direction);
    if (abs(denom) > RAY_TMIN) {
        let t = dot(Q.xyz - r.origin, n) / denom;
        if (t > RAY_TMIN && t < max) {
            let p = ray_at(r, t);
            let d = p - Q.xyz;
            let dot_uu = dot(u.xyz, u.xyz);
            let dot_uv = dot(u.xyz, v.xyz);
            let dot_vv = dot(v.xyz, v.xyz);
            let dot_du = dot(d, u.xyz);
            let dot_dv = dot(d, v.xyz);

            let inv_denom = 1.0 / (dot_uu * dot_vv - dot_uv * dot_uv);
            let u_coord = (dot_vv * dot_du - dot_uv * dot_dv) * inv_denom;
            let v_coord = (dot_uu * dot_dv - dot_uv * dot_du) * inv_denom;

            if (u_coord >= 0.0 && u_coord <= 1.0 && v_coord >= 0.0 && v_coord <= 1.0) {
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

fn hit_box(r: ray, center: vec3f, rad: vec3f, record: ptr<function, hit_record>, t_max: f32)
{
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
    return;
  }

  var t = tN;
  if (t < RAY_TMIN || t > t_max)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = -sign(r.direction) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  record.hit_anything = true;

  return;
}