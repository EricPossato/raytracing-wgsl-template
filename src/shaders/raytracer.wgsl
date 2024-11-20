const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_2_PI = 1.5707964f;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(0) @binding(1)
  var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> spheresb : array<sphere>;

@group(2) @binding(1)
  var<storage, read_write> quadsb : array<quad>;

@group(2) @binding(2)
  var<storage, read_write> boxesb : array<box>;

@group(2) @binding(3)
  var<storage, read_write> trianglesb : array<triangle>;

@group(2) @binding(4)
  var<storage, read_write> meshb : array<mesh>;

struct ray {
  origin : vec3f,
  direction : vec3f,
};

struct sphere {
  transform : vec4f,
  color : vec4f,
  material : vec4f,
};

struct quad {
  Q : vec4f,
  u : vec4f,
  v : vec4f,
  color : vec4f,
  material : vec4f,
};

struct box {
  center : vec4f,
  radius : vec4f,
  rotation: vec4f,
  color : vec4f,
  material : vec4f,
};

struct triangle {
  v0 : vec4f,
  v1 : vec4f,
  v2 : vec4f,
};

struct mesh {
  transform : vec4f,
  scale : vec4f,
  rotation : vec4f,
  color : vec4f,
  material : vec4f,
  min : vec4f,
  max : vec4f,
  show_bb : f32,
  start : f32,
  end : f32,
};

struct material_behaviour {
  scatter : bool,
  direction : vec3f,
};

struct camera {
  origin : vec3f,
  lower_left_corner : vec3f,
  horizontal : vec3f,
  vertical : vec3f,
  u : vec3f,
  v : vec3f,
  w : vec3f,
  lens_radius : f32,
};

struct hit_record {
  t : f32,
  p : vec3f,
  normal : vec3f,
  object_color : vec4f,
  object_material : vec4f,
  frontface : bool,
  hit_anything : bool,
};

fn ray_at(r: ray, t: f32) -> vec3f
{
  return r.origin + t * r.direction;
}

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray
{
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(cam.origin + offset, normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset));
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera
{
  var camera = camera();
  camera.lens_radius = aperture / 2.0;

  var theta = degrees_to_radians(vfov);
  var h = tan(theta / 2.0);
  var w = aspect_ratio * h;

  camera.origin = lookfrom;
  camera.w = normalize(lookfrom - lookat);
  camera.u = normalize(cross(vup, camera.w));
  camera.v = cross(camera.u, camera.w);

  camera.lower_left_corner = camera.origin - w * focus_dist * camera.u - h * focus_dist * camera.v - focus_dist * camera.w;
  camera.horizontal = 2.0 * w * focus_dist * camera.u;
  camera.vertical = 2.0 * h * focus_dist * camera.v;

  return camera;
}

fn environment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f
{
  var unit_direction = normalize(direction);
  var t = 0.5 * (unit_direction.y + 1.0);
  var col = (1.0 - t) * color1 + t * color2;

  var sun_direction = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  var sun_color = int_to_rgb(i32(uniforms[17]));
  var sun_intensity = uniforms[16];
  var sun_size = uniforms[18];

  var sun = clamp(dot(sun_direction, unit_direction), 0.0, 1.0);
  col += sun_color * max(0, (pow(sun, sun_size) * sun_intensity));

  return col;
}

fn check_ray_collision(r: ray, max: f32) -> hit_record
{
  var spheresCount = i32(uniforms[19]);
  var quadsCount = i32(uniforms[20]);
  var boxesCount = i32(uniforms[21]);
  var trianglesCount = i32(uniforms[22]);
  var meshCount = i32(uniforms[27]);

  var record = hit_record();
  var closest = max;
  record.t = max;
  record.hit_anything = false;
  var local_record = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);;



  for (var i = 0; i < spheresCount; i+=1){
    var sphere = spheresb[i];
    var center = vec3(sphere.transform.x, sphere.transform.y, sphere.transform.z);
    var radius = sphere.transform.w;
    
    if(hit_sphere(center, radius, r, &record, closest) && record.t < closest) {
      closest = record.t;  // Update closest intersection distance
      record.object_color = sphere.color;
      record.object_material = sphere.material;
    }
  }

    for (var i = 0; i < boxesCount; i+=1)
    {
      var box = boxesb[i];
      if (hit_box(r, box.center.xyz, box.radius.xyz, box.rotation.xyz, &local_record, closest) && local_record.t < closest) {
          closest = local_record.t; 
          record = local_record;     
          record.object_color = box.color;
          record.object_material = box.material;
        }
    }



  for (var i = 0; i < quadsCount; i+=1)
  {
    var quad = quadsb[i];
    if (hit_quad(r, quad.Q, quad.u, quad.v, &record, max) && record.t < closest){
      closest = record.t;
      record.object_color = quad.color;
      record.object_material = quad.material;
    };
  }

  record.frontface = dot(r.direction, record.normal) < 0.0;
  record.normal = select(-record.normal , record.normal,record.frontface);


  return record;
}

fn lambertian(normal: vec3f, absorption: f32, random_sphere: vec3f, rng_state: ptr<function, u32>) -> material_behaviour {
    // Get a random scatter direction by adding a random vector in a unit sphere to the normal
    var scatter_direction = normal + rng_next_vec3_in_unit_sphere(rng_state);

    // If the scatter direction is very small, reset it to the normal (to avoid degenerate rays)
    if (length(scatter_direction) < 1e-5) {
        scatter_direction = normal;
    }


    return material_behaviour(true, normalize(scatter_direction));
}

fn metal(normal : vec3f, direction: vec3f, fuzz: f32, random_sphere: vec3f) -> material_behaviour
{
  var after_reflect = reflect(normalize(direction), normal);
  var after_fuzz = after_reflect + fuzz * random_sphere;
  return material_behaviour(true, after_fuzz);
}

fn dielectric(normal: vec3f, r_direction: vec3f, refraction_index: f32, frontface: bool, random_sphere: vec3f, fuzz: f32, rng_state: ptr<function, u32>) -> material_behaviour {

    let ref_ratio = select(refraction_index, 1.0 / refraction_index, frontface);

    let unit_direction = normalize(r_direction);

    // Calculate cosine and sine of the angle between the ray and the normal
    var cos_theta = dot(-unit_direction, normal);
    cos_theta = min(cos_theta, 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);


    let not_refract = ref_ratio * sin_theta > 1.0;

    var r0 = (1.0 - ref_ratio) / (1.0 + ref_ratio);
    r0 = r0 * r0;
    let ref_prob = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

    // Decide whether to reflect or refract
    var direction: vec3f;
    if (not_refract || ref_prob > rng_next_float(rng_state)) {
        direction = reflect(unit_direction, normal);
    } else {
        let ref_perpendicular = ref_ratio * (unit_direction + cos_theta * normal);
        let ref_parallel = -sqrt(abs(1.0 - dot(ref_perpendicular, ref_perpendicular))) * normal;
        direction = ref_perpendicular + ref_parallel;
    }

    return material_behaviour(true, normalize(direction));
}

fn emmisive(color: vec3f, light: f32) -> material_behaviour
{
  return material_behaviour(false, color*light);
}

fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f {

    var max_bounces = i32(uniforms[2]);
    var accumulated_color = vec3f(1.0);
    var light_color = vec3f(0.0);
    var active_ray = r;

    var background_color1 = int_to_rgb(i32(uniforms[11]));
    var background_color2 = int_to_rgb(i32(uniforms[12]));

    for (var bounce = 0; bounce < max_bounces; bounce+=1) {

        var record = check_ray_collision(active_ray, RAY_TMAX);
        var smoothness = record.object_material.x; // =>0 -> metal, <0 -> dielectric
        var absorption = record.object_material.y; // = fuzz
        var specular = record.object_material.z;
        var light = record.object_material.w;

        var specular_prob = rng_next_float(rng_state);

        var behaviour : material_behaviour;


        if (!record.hit_anything) {
            light_color += accumulated_color * environment_color(active_ray.direction, background_color1, background_color2);
            break;
        }
        
        // emissive
        if (light > 0.0) {
            var emissive_behaviour = emmisive(record.object_color.xyz, light);
            light_color += accumulated_color * emissive_behaviour.direction;
            break;
        }
        if (smoothness < 0.0) {
            // Dielectric material
            var dielectric_response = dielectric(record.normal, active_ray.direction, specular, record.frontface, rng_next_vec3_in_unit_sphere(rng_state), absorption, rng_state); 
            behaviour = dielectric_response;
            record.p = record.p - 0.01 * record.normal;
        }
        else{
          // metal 
          if (specular_prob < specular) {
              var metal_response = metal(record.normal, active_ray.direction, absorption, rng_next_vec3_in_unit_sphere(rng_state));
              behaviour = metal_response;
          } else {
          // lambertian
            var lambertian_response = lambertian(record.normal, absorption, rng_next_vec3_in_unit_sphere(rng_state), rng_state);
            behaviour = lambertian_response;
          }
          accumulated_color *= mix(record.object_color.xyz,vec3f(1.0), specular); 
        }
        
        if (behaviour.scatter) {
            active_ray = ray(record.p, normalize(behaviour.direction));
            if (smoothness <= 0.0) {
                accumulated_color *= record.object_color.xyz;
            }
        } else {
            break;
        }
    }

    return light_color;
}



@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
    var rez = uniforms[1];
    var time = u32(uniforms[0]);

    // init_rng (random number generator) we pass the pixel position, resolution and frame
    var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), time);

    // Get uv
    var fragCoord = vec2f(f32(id.x), f32(id.y));
    var uv = (fragCoord + sample_square(&rng_state)) / vec2(rez);

    // Camera
    var lookfrom = vec3(uniforms[7], uniforms[8], uniforms[9]);
    var lookat = vec3(uniforms[23], uniforms[24], uniforms[25]);

    // Get camera
    var cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);
    var samples_per_pixel = i32(uniforms[4]);

    var color = vec3f(0.0);
    for (var i = 0; i < samples_per_pixel; i+=1) {
        let ray = get_ray(cam, uv, &rng_state);
        color += trace(ray, &rng_state);
    }

    color /= f32(samples_per_pixel);
    var frame_weight = 1.0 / f32(time);

    var color_out = vec4(linear_to_gamma(color), 1.0);
    var map_fb = mapfb(id.xy, rez);
    
    // 5. Accumulate the color
    var should_accumulate = uniforms[3];

    rtfb[map_fb] = rtfb[map_fb] * should_accumulate + color_out;
    fb[map_fb] = rtfb[map_fb] / rtfb[map_fb].w;
}