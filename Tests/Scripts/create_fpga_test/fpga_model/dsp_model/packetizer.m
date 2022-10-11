function [packet] = packetizer(cfg, nfo, detections, cfar_tresholds, cfar_data, blinder_data, sn_rem_data, dsp_in_data)
% PACKETIZER  Assemble the detection into packets
%
% [peaks, file_nfo] = peak_detection(data, cfg)
%
% IN
% cfg             : Configuration structure
% nfo             : Info structure
% detections      : Detectiosn structure form peak_detection
% cfar_data       : Sample data
% cfar_tresholds  : Possible selection for auxiliary data (default)
% blinder_data    : Possible selection for auxiliary data
% sn_rem_data     : Possible selection for auxiliary data
% dsp_in_data     : Possible selection for auxiliary data
%
% OUT
% packets   : Data packets (array of uint32)
%


   [~,~,endian] = computer;
   assert(endian == 'L', "Expecting little-endian")
   % if big-endian, probably needs some swapbytes()


   switch cfg.sample_select_ctrl.aux_select

      case 0
         aux = dsp_in_data;
      case 1
         aux = sn_rem_data;
      case 2
         aux = blinder_data;
      case 3
         aux = cfar_tresholds;
      otherwise
         error('Unknown cfg.sample_select_ctrl.aux_select: %d', cfg.sample_select_ctrl.aux_select);
   end


   assert(cfg.sample_select_ctrl.n_neighbors>0)
   if cfg.sample_select_ctrl.n_neighbors > 15
      warning(sprintf('cfg.sample_select_ctrl.n_neighbors is out-of-spec: %d\n', cfg.sample_select_ctrl.n_neighbors));
   end



   % Header (included by default)
   if isfield(cfg.sample_select_ctrl, 'header') && cfg.sample_select_ctrl.header == 0
      packet = []
   else
      packet = gen_header(cfg, nfo, detections, cfar_data);
   end


   % Detections
   for i=1:cfg.sample_select_ctrl.n_detections
      if i <= length(detections)
         curr = gen_detection(cfg, detections(i), cfar_data, aux);
      else
         % Fill empty detections with 0
         curr  = uint32( zeros(1, detection_len(cfg)) );
      end

      packet = [packet curr];
   end


   % Raw Data
   if cfg.sample_select_ctrl.select_mode == 0 && cfg.sample_select_ctrl.all_samples ~= 0

      raw = cfar_data;

      if cfg.sample_select_ctrl.all_aux ~= 0
         raw = [raw aux];
      end

      packet = [packet typecast(int16(raw), 'uint32')'];
   end


   % Debug
   % fprintf(hex_dump(packet));


endfunction



function [hdr] = gen_header(cfg, nfo, detections, data)
% GEN_HEADER   Generate the header section and return it as an array of uint32

   ID_COMPRESS    = 4;
   ID_CLOUD       = 5;
   ID_CLOUD_EXT   = 6;


   % First dword of the header give the general format of the packets
   fmt=struct;

   if cfg.sample_select_ctrl.select_mode == 1
      fmt.id         = ID_COMPRESS;
      fmt.n_sample   = 2 * cfg.sample_select_ctrl.n_neighbors + 1;

   else

      if cfg.sample_select_ctrl.all_samples ~= 0 && cfg.sample_select_ctrl.all_aux ~= 0
         fmt.id         = ID_CLOUD_EXT;
         fmt.n_sample   = length(data);   % smpl + aux


      elseif cfg.sample_select_ctrl.all_samples ~= 0
         fmt.id         = ID_CLOUD;
         fmt.n_sample   = length(data);

      else
         fmt.id         = ID_CLOUD;
         fmt.n_sample   = 0;

      end
   end


   fmt.version    = 1;
   fmt.m_detect   = cfg.sample_select_ctrl.n_detections;
   fmt.detect_cnt = length(detections);


   if nfo.leftovers ~= 0
      leftovers = 1;
   else
      leftovers = 0;
   end


   % bit packing

   hdr = uint32(zeros(1,5));

   hdr(1) = uint32( ...
               bitand(fmt.detect_cnt, 0x3F) * 2**26 + ...
               bitand(fmt.n_sample, 0xFFF) * 2**14 + ...
               bitand(fmt.m_detect, 0x3F) * 2**8 + ...
               bitand(fmt.version, 0xF) * 2**4 + ...
               bitand(fmt.id, 0xF) ...
            );

   hdr(2) = uint32(
               leftovers * 2**31 + ...
               bitand(nfo.angpos_laser, 0X1FF) * 2**22 + ...
               bitand(nfo.pd_number, 0X3F) * 2**16 + ...
               bitand(nfo.frame_number, 0XFFFF) ...
            );

   hdr(3) = uint32(nfo.timestamp);
   hdr(4) = typecast(uint8([nfo.cfg_id nfo.frame_id, nfo.optical_id, nfo.acq_id]), 'uint32');
   hdr(5) = typecast([...
               uint16(nfo.noise_std_dev) ...
               typecast(int16(nfo.baseline),'uint16') ...
            ], 'uint32');

endfunction


function len = detection_len(cfg)
% DETECTION_LEN Number of uint32 for each detection section
   len = 4;

   if cfg.sample_select_ctrl.select_mode == 1
      len += 2*cfg.sample_select_ctrl.n_neighbors+1;  % smpl + aux
   end

endfunction


function [det] = gen_detection(cfg, detection, data, thld)
% GEN_DETECTION   Generate a detection section and return it as an array of uint32

   assert(length(data) == length(thld), "Data and Threshold length mismatch")

   det = uint32( zeros(1, detection_len(cfg)) );


   % Include shot waveform for "compress" mode
   if cfg.sample_select_ctrl.select_mode == 1

      % When detection is close to an edge some care must be taken to keep short waveform length constant
      len      = 2*cfg.sample_select_ctrl.n_neighbors + 1;
      assert(length(data) >= len, "Not enough data for short waveform")

      limit_hi = max(0, length(data) - len);
      offset   = max(0, min(limit_hi, round(detection.bin) - cfg.sample_select_ctrl.n_neighbors));

      shortwave = typecast(int16([ ...
                     data(offset+1 : offset+len)
                     thld(offset+1 : offset+len)
                  ]), 'uint32');

      assert(length(shortwave) == len)

      det(5:end) = shortwave;
   else
      offset = 0;
   end



   % Common to all detections
   fmt = peak_detection_format();

   det(1) = typecast([ ...
               typecast(uint16(detection.bin * 2**fmt.bin.fractional),'int16') ...  % bin (interpolated)
               int16(detection.smpl_before) ...
            ],'uint32');

   det(2) = typecast(int16([ ...
               detection.bl_before ...
               detection.bl_after ...
            ]),'uint32');


   if detection.saturated == 1
      % For saturated pulse replace magnitude field with pulse length
      assert(detection.pulse_len > 0 && detection.pulse_len < 1024)
      det(3) = typecast(int32(detection.pulse_len),'uint32');
   else
      % Magnitude field
      det(3) = typecast(int32(detection.mag),'uint32');
   end


   assert(detection.saturated == 0 || detection.saturated == 1)

   det(4) = typecast(uint16([
            (detection.saturated * 2**15 + detection.idx) ...
            offset ...
            ]),'uint32');

endfunction



function [str] = hex_dump(data, width=8, sep=' ')
% HEX_DUMP  Display data content in hexadecimal
%
   assert(class(data) == 'uint32')

   str = "";

   for offset=[1:width:length(data)]
      % subset = data(offset : offset+width-1);
      ln = sprintf("%08X:", offset-1);
      for i=[0:width-1]
         ptr = offset+i;
         if ptr > length(data)
            break;
         end
         ln = [ln sprintf("%s%08X", sep, data(ptr))];
      end
      ln = [ln sprintf("\n")];
      str = [str ln];
   end
endfunction


% vim:tw=0:ts=3:sts=3:sw=3:et
