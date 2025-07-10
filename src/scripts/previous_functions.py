def generate_texture_old(batch, args):

    sa_iter = args.sa_iter
    loops_number = args.loop_iter

    OUTDIR = args.out_dir

    config = args.config


    # Encode
    with (torch.no_grad()):

        OUTDIR_f = os.path.join(OUTDIR, batch['fname'][0][8:])

        text = batch['text']
        text = ' '.join(text[0].replace("\n", "").split('_'))

        sd_cfg.txt2img.prompt = "turn around, " + text + " , high quality, high resolution"
        sd_cfg.inpaint.prompt = "turn around, " + text + " , high quality, high resolution"

        # Check if OUTDIR exists
        if not os.path.exists(OUTDIR_f):
            os.makedirs(OUTDIR_f)

        from pytorch3d.io import load_obj

        batch['tex'] = batch['tex'].to(net.dtype)
        batch['nn_ids'] = batch['nn_ids'].long().to(net.dtype)
        batch['nn_dists'] = batch['nn_dists'].to(net.dtype)
        batch['nn_gamma'] = batch['nn_gamma'].to(net.dtype)
        batch['nn_theta'] = batch['nn_theta'].to(net.dtype)
        batch['nn_texels_id'] = batch['nn_texels_id'].to(net.dtype)
        batch['gen_images'] = batch['gen_images'].to(net.dtype)
        batch['cam_params'] = batch['cameras_dict']
        batch['text'] = batch['text']
        # verts_or = batch['mesh']['verts'].clone()
        import copy

        mesh_dict = copy.deepcopy(batch['mesh'])
        for idx in range(3):
            # Initialize camera params
            if idx == 0:
                batch['mesh']['verts'] = normalize_mesh(batch['mesh']['verts'][0], target_scale=0.6,
                                                        mesh_dy=0.25, mean=None).unsqueeze(
                    0)  # tri_verts_mean.float()
                batch['mesh'] = init_mesh_ours(batch['mesh'], batch['tex'].permute(0, 3, 1, 2),
                                               batch['mesh']['verts'])
            else:
                # packed_verts = verts_or.to(batch['mesh'].verts_packed().device)  # Access packed vertices

                mesh_verts = normalize_mesh(mesh_dict['verts'][0], target_scale=0.6,
                                            mesh_dy=0.25, mean=None).unsqueeze(
                    0)  # tri_verts_mean.float()
                batch['mesh'] = init_mesh_ours(mesh_dict,
                                               torch.clip(tex_synth_print.to(mesh_dict['verts'].device), min=0,
                                                          max=1).float(),
                                               mesh_verts)

            if idx == 0:
                camera_params = {}
                camera_params["dist"] = [1.5 for _ in range(2)]
                camera_params["elev"] = [25 for _ in range(2)]
                camera_params["azim"] = [((360 / 2) * i)
                                         for i in range(2)]
                # camera_params["azim"] = [0, 45, 180, 315]

                mult = 1
            if idx == 1:
                camera_params = {}
                camera_params["dist"] = [1.3 for _ in range(4)]
                camera_params["elev"] = [25 for _ in range(4)]
                camera_params["azim"] = [((360 / 4) * i)
                                         for i in range(4)]
                mult = 2
                # print('extra views = ', config.extra_views)
            if idx == 2:
                camera_params = {}
                camera_params["dist"] = [1.3 for _ in range(6)]
                camera_params["elev"] = [25 for _ in range(6)]
                camera_params["azim"] = [((360 / 6) * i)
                                         for i in range(6)]
                mult = 3

            latents_list = []
            depth_list = []
            masked_list = []

            for iter, (d, e, a) in enumerate(
                zip(camera_params['dist'], camera_params['elev'], camera_params['azim'])):
                # if idx == 0:
                cameras = init_camera({'dist': d, 'elev': e, 'azim': a}, config.render_size, DEVICE, paint3d=True)
                cameras = cameras[iter]
                # else:
                #     cameras = init_camera({'dist': d, 'elev': e, 'azim': a}, config.render_size, DEVICE,
                #                           paint3d=False)

                renderer = init_renderer(cameras,
                                         shader=init_soft_phong_shader(
                                             camera=cameras,
                                             # blend_params=BlendParams(),
                                             device=DEVICE),
                                         image_size=config.render_size,
                                         faces_per_pixel=1
                                         ).to(DEVICE)
                if idx == 1 or idx == 2:
                    mesh_verts = normalize_mesh(mesh_dict['verts'][0], target_scale=0.6,
                                                mesh_dy=0.25, mean=None).unsqueeze(
                        0)  # tri_verts_mean.float()

                    blend_global_erod = erode_mask(blend_global.to(mesh_dict['verts'].device))
                    mesh_masked = init_mesh_ours(mesh_dict,
                                                 torch.clip(blend_global_erod.permute(0, 2, 3, 1), min=0,
                                                            max=1).float(),
                                                 mesh_verts)

                    masked_img, _ = renderer(mesh_masked.to('cuda'), camera=cameras)
                    masked_list.append(masked_img[..., [0]])
                    save_image(masked_img[..., [0]].permute(0, 3, 1, 2), 'depth_' + str(iter) + '.png')
                    save_image(tex_synth_print.permute(0, 3, 1, 2), 'rgb_' + str(iter) + '.png')

                    # save_image(blend_global, 'blend_glob_' + str(iter) + '.png')

                    # exit()

                latents, fragments = renderer(batch['mesh'].to('cuda'), camera=cameras)  # image: (N, H, W, C)
                save_image(latents.permute(0, 3, 1, 2), 'latents_' + str(iter) + '.png')
                # background_mask_list.append(fragments.zbuf.repeat(1,1,1,3))

                latents_list.append(latents[..., :3])

                try:
                    absolute_depth, relative_depth = get_relative_depth_map(fragments.zbuf)
                except:
                    continue

                save_image(relative_depth.unsqueeze(0), 'depth_' + str(iter) + '.png')

                depth_list.append(relative_depth.unsqueeze(-1))  # -1 - 1

            if idx == 0:
                latents = torch.cat((latents_list[0], latents_list[1]), dim=2).permute(0, 3, 1, 2)
                sd_cfg.txt2img.height = 512
            elif idx == 1:
                latents = torch.cat((latents_list[2], latents_list[3]), dim=2).permute(0, 3, 1, 2)
                masked_img_mesh = torch.cat((masked_list[2], masked_list[3]), dim=2).permute(0, 3, 1, 2)
                masked_img_mesh = F.interpolate(masked_img_mesh, size=(512 * 1, 512 * 2))
                sd_cfg.txt2img.height = 512
            elif idx == 2:
                sd_cfg.txt2img.height = 512
                latents = torch.cat((latents_list[4], latents_list[5]), dim=2).permute(0, 3, 1, 2)
                masked_img_mesh = torch.cat((masked_list[4], masked_list[5]), dim=2).permute(0, 3, 1, 2)
                masked_img_mesh = F.interpolate(masked_img_mesh, size=(512 * 1, 512 * 2))
                sd_cfg.txt2img.height = 512

            latents = F.interpolate(latents, size=(512 * 1, 512 * 2))

            if idx == 0:
                rel_depth_normalized = torch.cat((depth_list[0], depth_list[1]), dim=2).permute(0, 3, 1, 2)

            elif idx == 1:
                rel_depth_normalized_input = torch.cat((depth_list[2].clone(), depth_list[3].clone()),
                                                       dim=2).permute(0, 3, 1, 2)
                rel_depth_normalized_input = F.interpolate(rel_depth_normalized_input, (512 * 1, 512 * 2))
                rel_depth_normalized = torch.cat(
                    (torch.cat((depth_list[0], depth_list[1]), dim=2),
                     torch.cat((depth_list[2], depth_list[3]), dim=2)),
                    # torch.cat((depth_list[4], depth_list[5]), dim=2)),
                    dim=1).permute(0, 3, 1, 2)

            elif idx == 2:
                rel_depth_normalized = torch.cat(
                    (torch.cat((depth_list[0], depth_list[1]), dim=2),
                     torch.cat((depth_list[2], depth_list[3]), dim=2),
                     torch.cat((depth_list[4], depth_list[5]), dim=2)),
                    dim=1).permute(0, 3, 1, 2)
                rel_depth_normalized_input = torch.cat((depth_list[4], depth_list[5]), dim=2).permute(0, 3, 1, 2)
                rel_depth_normalized_input = F.interpolate(rel_depth_normalized_input, (512 * 1, 512 * 2))

            rel_depth_normalized = F.interpolate(rel_depth_normalized, size=(512 * mult, 512 * 2))
            if idx == 0:
                sd_cfg.denoising_strength = 1.
            elif idx == 1:
                sd_cfg.denoising_strength = 1.
            elif idx == 2:
                sd_cfg.denoising_strength = 1.


            rel_depth_normalized = F.interpolate(rel_depth_normalized, (512 * mult, 512 * 2))
            rel_depth_normalized = rearrange(rel_depth_normalized, 'b c h (n w) -> (n b) c h w', n=2)
            rel_depth_normalized = rearrange(rel_depth_normalized, 'b c (n h) w -> (n b) c h w', n=mult)

            if idx == 0:
                from torchvision.io import read_image
                import objaverse
                uid_full = ['/'.join(f.split('/')[-2:]).split('.')[0] for f in
                            objaverse.load_objects(uids=[batch['fname'][0][8:]]).values()]

                # Load tex
                outputs_noise = read_image(os.path.join(paint3D_dir, uid_full[0], 'init-img-0.png')) / 255.

                outputs_noise = outputs_noise.unsqueeze(0)
                outputs_noise = rearrange(outputs_noise, 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise = rearrange(outputs_noise, 'b c (n h) w -> (n b) c h w', n=1)

                outputs_noise = outputs_noise

                paint_3d_flag = True
            elif idx == 1:
                # Load init views
                outputs_noise = read_image(os.path.join(paint3D_dir, uid_full[0], 'init-img-0.png')) / 255.

                outputs_noise = outputs_noise.unsqueeze(0)
                outputs_noise = rearrange(outputs_noise, 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise = rearrange(outputs_noise, 'b c (n h) w -> (n b) c h w', n=1)
                paint_3d_flag = True

                inpainted_images = inpaint_viewpoint(
                    sd_cfg=sd_cfg,
                    cnet=inpaint_cnet,
                    save_result_dir=OUTDIR_f,
                    latent_im=latents,
                    masked_img_mesh=masked_img_mesh,
                    depth_im=rel_depth_normalized_input,
                    idx=idx,
                    inpaint_views=["2_3"]
                )[0]
                inpainted_images = transforms.ToTensor()(inpainted_images)

                inpainted_images = rearrange(inpainted_images.unsqueeze(0), 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise = torch.cat((outputs_noise, inpainted_images), dim=0)

            elif idx == 2:
                # Load tex
                outputs_noise = read_image(os.path.join(paint3D_dir, uid_full[0], 'init-img-0.png')) / 255.
                # outputs_noise = read_image(os.path.join(paint3D_dir, uid_full[0][8:], 'init-img-0.png')) / 255.

                outputs_noise = outputs_noise.unsqueeze(0)
                outputs_noise = rearrange(outputs_noise, 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise = rearrange(outputs_noise, 'b c (n h) w -> (n b) c h w', n=1)
                # outputs_noise2 = read_image(os.path.join(paint3D_dir, uid_full[0], 'view_5_6_rgb_inpaint_0.png')) / 255.
                outputs_noise2 = read_image(os.path.join(OUTDIR_f, 'view_2_3_rgb_inpaint_0.png')) / 255.

                outputs_noise2 = outputs_noise2.unsqueeze(0)
                outputs_noise2 = rearrange(outputs_noise2, 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise2 = rearrange(outputs_noise2, 'b c (n h) w -> (n b) c h w', n=1)
                inpainted_images = inpaint_viewpoint(
                    sd_cfg=sd_cfg,
                    cnet=inpaint_cnet,
                    save_result_dir=OUTDIR_f,
                    latent_im=latents,
                    masked_img_mesh=masked_img_mesh,
                    depth_im=rel_depth_normalized_input,
                    idx=idx,
                    inpaint_views=["2_3"]
                )[0]
                inpainted_images = transforms.ToTensor()(inpainted_images)

                inpainted_images = rearrange(inpainted_images.unsqueeze(0), 'b c h (n w) -> (n b) c h w', n=2)
                outputs_noise = torch.cat((outputs_noise, outputs_noise2, inpainted_images), dim=0)

            outputs_noise = outputs_noise.to(DEVICE) * (rel_depth_normalized > 0) + torch.ones(
                outputs_noise.shape).to(DEVICE) * (rel_depth_normalized == 0)

            batch['mesh'].textures = TexturesUV(
                maps=torch.clip(batch['tex'].to(batch['mesh'].device), min=0, max=1).float(),  # B, H, W, C
                faces_uvs=[batch['mesh'].textures.faces_uvs_padded()[0]],
                verts_uvs=[batch['mesh'].textures.verts_uvs_padded()[0]],
                sampling_mode="bilinear")

            print('outputs_noise shape: ', outputs_noise.shape)

            # Render Views Normals and
            images_set, _ = render_depth_normals_tex(batch['mesh'].to('cuda'),
                                                     torch.ones(batch['tex'].shape).to('cuda'),
                                                     torch.ones(batch['tex_hd'].shape).to('cuda'), camera_params,
                                                     1024,
                                                     1, args.cross_attention_window, 'cuda',
                                                     gen_images=outputs_noise.unsqueeze(0), up=True,
                                                     # blend_tex_size=1024, paint3d=paint_3d_flag)
                                                     blend_tex_size=1024, paint3d=paint_3d_flag,
                                                     number_of_views=mult * 2)

            print('images set features shape: ', images_set['tex_features'].shape)
            if images_set['tex_features'].shape[1] == 256:
                loc_map = images_set['uv_loc']
                normal_map = images_set['uv_normals']
                tex_input = batch['tex']
                active_texels_input = images_set['active_texels']
            else:
                loc_map = images_set['uv_loc_1024']
                normal_map = images_set['uv_normal_1024']
                tex_input = batch['tex_hd']
                active_texels_input = images_set['active_texels_1024']

            tex_real, tex_real_hd, tex, tex_features, active_texels, active_view_texels, active_tokens = net.get_input(
                tex_input.to('cuda'),
                tex_input.to('cuda'),
                images_set['tex_features'].to('cuda'),
                active_texels_input.to('cuda'),
                loc_map.to('cuda'),
                normal_map.to('cuda'),
                args.cross_attention_window,
                torch.float32)


            print('init inputs shapes: ', tex_features.shape, active_view_texels.shape,
                  images_set['uv_loc_1024'].shape, images_set['uv_normal_1024'].shape, tex.shape)
            # Initialize inputs
            tex, tex_features, active_view_texels, loc, normals, tex_geom = net.init_inputs(tex_features,
                                                                                            active_view_texels,
                                                                                            loc_map.to(
                                                                                                tex.device),
                                                                                            normal_map.to(
                                                                                                tex.device),
                                                                                            tex)

            tex_blend, blend_mask = net.blending_model.blend_forward(tex, tex_features, tex_geom,
                                                                     active_view_texels,
                                                                     active_tokens, texture_size=1024)

            empty_texels = tex_blend[torch.logical_not(blend_mask)]
            empty_texels[:] = 0.

            tex_blend[torch.logical_not(blend_mask)] = empty_texels

            tex_synth = tex_blend.clone()

            import objaverse

            uid_full = ['/'.join(f.split('/')[-2:]).split('.')[0] for f in
                        objaverse.load_objects(uids=[batch['fname'][0][8:]]).values()]

            # Load tex
            blend_global = blend_mask.reshape(1, 1, tex_input.shape[2], tex_input.shape[2]).float()

            if args.rgb_space:
                tex_synth_print = tex_synth.reshape(tex_input.shape)
                if idx != 2:
                    tex_synth_print = dilate_mask(tex_synth_print.float().permute(0, 3, 1, 2), iter=8).permute(0, 2,
                                                                                                               3, 1)


                mask_texels_empty_ = torch.all(tex_synth_print == 0, dim=-1)
                empty_texels = tex_synth_print[
                    mask_texels_empty_]  # tex_synth_print[torch.logical_not(blend_global).permute(0,2,3,1).squeeze(-1)]
                empty_texels[:] = 1.
                empty_texels[..., 1] = 51 / 255
                tex_synth_print[mask_texels_empty_] = empty_texels

                tex_blend_print = tex_blend[0].reshape(tex_input.shape)
            else:
                tex_synth_print = tex_synth.reshape(tex_input.shape)
                tex_blend_print = tex_blend.reshape(tex_input.shape)

            _, batch['mesh'] = render_multi_inference(batch['mesh'].to(tex.device), tex_synth_print.to(tex.device),
                                                      renderer,
                                                      outputs_noise.permute(0, 2, 3, 1).to(tex.device), 512,
                                                      out_dir=os.path.join(OUTDIR_f,
                                                                           str(idx).zfill(5)), loop_idx=idx)

            batch['mesh'].textures = TexturesUV(
                maps=torch.clip(tex_synth_print.to(tex.device), min=0, max=1).float(),  # B, H, W, C
                faces_uvs=[batch['mesh'].textures.faces_uvs_padded()[0]],
                verts_uvs=[batch['mesh'].textures.verts_uvs_padded()[0]],
                sampling_mode="bilinear")
