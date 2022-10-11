function data = get_raw_data(filename,start_trace=1,num_trace=0,output_files_types={'dat'; 'csv'})
%
% function d = get_raw_data(filename)
% get_raw_data: Extracts the data from the software's raw acquisition files and generates .csv and .dat files
% start_trace: Indicate at which trace to start capturing the data
% num_trace: Indicated the number of trace to capture (capture all trace when num_trace=0)
% get_raw_data: extracts the data from the software's raw acquisition files and generates .csv and .dat files
%
% ex:
%   d = get_raw_data('LSP_Acquisition_Get_raw_waveform_3D_scan_ROI_0_Config_0.txt');
%
  pkg load io

  frame_index = "{'frame_cfg_index': ";
  optical_tile_idx = "{'optical_tile_index': ";
  acq_tile_idx = "{'acquisition_tile_index': ";

	data_start_pattern = "{'waveform_data': [";
	data_header_pattern = "'waveform_header': {";

  gathering_cfg = 0;
  f = fopen(filename, 'r');

% read file into structures / arrays
  while feof(f)==0

	  raw_line = fgetl(f);

%% gather structure information
% needs to be this way beacause of the file structure which has the same fields for at different locations
% so we make sure these thress lines follow each other before adding configuration
    switch gathering_cfg
      case 1
        [~, ~, ~, ~, o_idx, ~, ~] = regexp(raw_line, [".*" optical_tile_idx "(\\d+)}"]);
        if( ~isempty(o_idx) )
          gathering_cfg = 2;
        else
          gathering_cfg = 0;
        endif;
      case 2
         [~, ~, ~, ~, a_idx, ~, ~] = regexp(raw_line, [".*" acq_tile_idx "(\\d+)}"]);
         gathering_cfg = 0;
         if( ~isempty(a_idx) )
           frame_idx = str2double(f_idx{1}{1}) + 1;
           tile_idx = str2double(o_idx{1}{1}) + 1;
         endif;
      otherwise
        [~, ~, ~, ~, f_idx, ~, ~] = regexp(raw_line, [".*" frame_index "(\\d+)}"]);
        if( ~isempty(f_idx) )
          gathering_cfg = 1;
        endif
    endswitch

    acq_cfg = regexp(raw_line, '.*acq_tiles_cfg\.(?<field>\D+).:.(?<val>\d+)}', 'names');
    if( isempty(acq_cfg)==0 )
      eval(['data{', num2str(frame_idx), ',', num2str(tile_idx), '}.cfg.', acq_cfg.field, '=', acq_cfg.val,';']);
    endif


%% acquisition capture header
    acq_head_pos = strfind(raw_line, data_header_pattern);
    if( acq_head_pos>0 )
      pos = acq_head_pos+numel(data_header_pattern)-2;
      acq_head = regexp(raw_line(pos:end), '[ ^{]''(?<field>[\D_]+)'':.(?<val>\d+)[,}]', 'names');
      if( isempty(acq_head)==0 )
        for n=1:numel(acq_head)
          eval(['header.', acq_head(n).field, '=', acq_head(n).val,';']);
        endfor
        tile_idx = header.roi_optical_tile_index + 1;
        data{frame_idx, tile_idx}.header = header;
      endif
    endif;


%% Data
    data_start_pos = strfind(raw_line, data_start_pattern);
    if( isempty(data_start_pos)==0 )

      angles   = data{frame_idx, tile_idx}.cfg.VSeg;
      lasers   = data{frame_idx, tile_idx}.cfg.HSeg;
      N        = data{frame_idx, tile_idx}.cfg.BasePoints;
      oversamp = data{frame_idx, tile_idx}.cfg.Oversampling;
      
      samples = prod([angles, lasers, N, oversamp]);

      mess = sprintf('frame=%u, tile=%u, angles=%d, lasers=%u, N=%u and total samples=%u\n', frame_idx-1, tile_idx-1, angles, lasers, N, samples);
%      mess = sprintf('frame=%u, tile=%u, angles=%d, lasers=%u, N=%u and total samples=%u\n', frame_idx-1, tile_idx-1, angles, lasers, N, data{frame_idx, tile_idx}.header.sample_qty);
%      if( prod([angles, lasers, N, oversamp])~=data{frame_idx, tile_idx}.header.sample_qty )
%        error(['Configuration is wrong; '], mess);
%      else
%        fprintf(['Configuration: ', mess]);
%      endif

  % Arrange data
      d0 = textscan(raw_line(data_start_pos+numel(data_start_pattern):end-2),'%f', 'delimiter', ',');
      d1 = (d0{1}(1:2:end-1) + d0{1}(2:2:end)*2^8)(1:samples);

      % unsigned to signed
      p = find(d1>2^15-1);
      d1(p) = d1(p) - 2^16;
      
    % to columns (not required but nice if we want to use in Octave)
      d = reshape(d1(1:angles*lasers*N*oversamp), N*oversamp, []);
      if( ~isfield(data{frame_idx, tile_idx}, 'raw') )
        acq_idx = 1;
      else
        acq_idx = numel(data{frame_idx, tile_idx}.raw)+1;
      endif;
      data{frame_idx, tile_idx}.raw{acq_idx} = d;

      clear d0 d1 d raw_line;
  	endif
  endwhile

  fclose(f);

  if numel(output_files_types)==0
    return;
  endif;

%% Data file creation
  output_files_types = lower(output_files_types);
  dat = cell2mat(strfind(output_files_types, 'dat'));
  csv = cell2mat(strfind(output_files_types, 'csv'));
  xlsx = cell2mat(strfind(output_files_types, 'xls'));

  if dat+csv+xlsx==0
    error('Unknown files type given; no output files generated.\n');
  endif;

  fprintf("Creating files (this takes a while) ...");

  [nb_frames, nb_tiles] = size(data);
  for frame=1:nb_frames
    fprintf('{%u} ', frame-1);
    tiles = find(~cellfun(@isempty, data(frame, :)));
    if( ~isempty(tiles) )
      dir = sprintf('tile_%u', frame-1);
      mkdir(dir);
      for tile=tiles
        fprintf('[%u] ', tile-1);
        if( isfield(data{frame,tile}, 'raw') )

            nb_acq = numel(data{frame,tile}.raw);
            for acq=1:nb_acq
                s = size(data{frame,tile}.raw{acq});
            % If num_trace higher than 0, only capture [start_trace:start_trace+num_trace-1] range  
            if (num_trace > 0)
               s(2) = start_trace+num_trace-1;
            endif 


        %% create files
            if xlsx>0
              fprintf(".");
              xlswrite(sprintf('%s/raw_data_%u', dir, tile-1), data{frame,tile}.raw{acq}, acq);
             endif

            if csv>0
              f = fopen(sprintf('%s/raw_data_%u_%u.csv', dir, tile-1, acq-1), 'w');
              for k=start_trace:s(2)
                tr = ones(s(1), 1)*(k-1);
                dt = reshape([tr, data{frame,tile}.raw{acq}(:,k)], [], 2)'(:);
                fprintf(f, "%u;%u\n", dt);
              endfor
              fclose(f);
            endif

            if dat>0
              t = 0;
              freq = 160.0e6;
              dat_head = [
                           '%time i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_sig i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_val ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_blafter i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_blbefore ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_noise i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_sig ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_val i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_noise ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_sig i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_val ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_detind i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_sig ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_threshold i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_val ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_bin i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_mag ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_val i_lca3_fpga_sys.i_dsp_fpga1.dsp4_sati_bldiff ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_bin i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_mag ', ...
                           'i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_val'
                         ]';

              f = fopen(sprintf('%s/raw_data_%u_%u.dat', dir, tile-1, acq-1), 'w');
              fprintf(f, "%s\n", dat_head(:));
              for k=start_trace:s(2)
                fprintf(f, '%g %u %u 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n', [t, 0, 0]);
                t = t + 1/freq;
                time = [1:s(1)]'.*t;
                valid =ones(s(1), 1);
                dt = reshape([time, data{frame,tile}.raw{acq}(:,k), valid], [], 3)'(:);
                fprintf(f, '%g %u %u 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n', dt);
                fflush(f);
              endfor
              fclose(f);
            endif

            fprintf(".");
            if mod(acq,16)==0 || acq==nb_acq
              fprintf('%u', acq);
            endif

          endfor
        endif
      endfor
    endif
  endfor

  fprintf(" *done.\n\n");

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et