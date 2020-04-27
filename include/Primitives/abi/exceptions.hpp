#pragma once

namespace glasssix::exposing
{
	struct glasssix_abi_error
	{
	};

	struct glasssix_abi_no_interface : glasssix_abi_error
	{
	};

	struct glasssix_abi_not_implemented : glasssix_abi_error
	{
	};
}
