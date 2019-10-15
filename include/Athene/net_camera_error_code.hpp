#pragma once

namespace glasssix
{
	namespace hippogriff
	{
		enum class net_camera_error_code
		{
			codec_failure,
			timeout,
			reading_packet_failure,
			stream_index_mismatch,
			decode_failure,
			allocating_sws_context_failure,
			sws_scaling_failure,
			packet_clone_failure,
			unsupported_format,
			other
		};
	}
}
