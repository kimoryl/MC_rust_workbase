// Clean-room thruster exhaust plume and aerodynamic energy fragment shader
// for Horizon RenderPipeline (linear light output, no custom bindings).
fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    let radial = abs(input.uv.x * 2.0 - 1.0);
    let axial = clamp(input.uv.y, 0.0, 1.0);

    // Nozzle expansion profile narrows smoothly toward the tail.
    let half_width = mix(0.88, 0.16, axial);
    let cross_mask = clamp(1.0 - radial / max(half_width, 0.05), 0.0, 1.0);
    let radial_profile = cross_mask * cross_mask;

    // Animated Mach shock-diamond standing wave along exhaust axis.
    let wave_phase = axial * 18.849556 - frame.time * 26.0;
    let mach_diamond = 0.65 + 0.35 * sin(wave_phase);
    let flicker = 0.92 + 0.08 * sin(frame.time * 43.0 + input.position.x * 2.3 + input.position.z * 1.9);

    // Axial attenuation from white-hot nozzle core to soft plasma tail.
    let axial_fade = pow(clamp(1.0 - axial, 0.0, 1.0), 1.35);
    let core_intensity = clamp(radial_profile * axial_fade * mach_diamond * flicker, 0.0, 1.0);

    // Hot white-core boost blended with the quad's linear tint color.
    let hot_core = pow(core_intensity, 2.2);
    let emission = input.color.rgb * (0.85 + 1.35 * core_intensity) + vec3<f32>(0.9, 0.95, 1.0) * (0.75 * hot_core);
    let alpha = clamp(input.color.a * core_intensity, 0.0, 1.0);

    return vec4<f32>(emission, alpha);
}
