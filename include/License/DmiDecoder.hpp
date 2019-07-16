#ifndef _DMIDECODER_HPP_
#define _DMIDECODER_HPP_
#ifdef __GNUC__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "singleton.hpp"

#include <exception>
#include <string>
#include <unordered_map>

namespace glasssix
{
	namespace hippogriff
	{

#define SUPPORTED_SMBIOS_VER 0x030200
#define FLAG_NO_FILE_OFFSET     (1 << 0)
#define FLAG_STOP_AT_EOT        (1 << 1)

#define SYS_FIRMWARE_DIR "/sys/firmware/dmi/tables"
#define SYS_ENTRY_FILE SYS_FIRMWARE_DIR "/smbios_entry_point"
#define SYS_TABLE_FILE SYS_FIRMWARE_DIR "/DMI"

		typedef unsigned char u8;
		typedef unsigned short u16;
		typedef signed short i16;
		typedef unsigned int u32;

		/*
		* You may use the following defines to adjust the type definitions
		* depending on the architecture:
		* - Define BIGENDIAN on big-endian systems.
		* - Define ALIGNMENT_WORKAROUND if your system doesn't support
		*   non-aligned memory access. In this case, we use a slower, but safer,
		*   memory access method. This should be done automatically in config.h
		*   for architectures which need it.
		*/

#ifdef BIGENDIAN
		typedef struct {
			u32 h;
			u32 l;
		} u64;
#else
		typedef struct {
			u32 l;
			u32 h;
		} u64;
#endif

#if defined(ALIGNMENT_WORKAROUND) || defined(BIGENDIAN)
		static inline u64 U64(u32 low, u32 high)
		{
			u64 self;

			self.l = low;
			self.h = high;

			return self;
		}
#endif

		/*
		* Per SMBIOS v2.8.0 and later, all structures assume a little-endian
		* ordering convention.
		*/
#if defined(ALIGNMENT_WORKAROUND) || defined(BIGENDIAN)
#define WORD(x) (u16)((x)[0] + ((x)[1] << 8))
#define DWORD(x) (u32)((x)[0] + ((x)[1] << 8) + ((x)[2] << 16) + ((x)[3] << 24))
#define QWORD(x) (U64(DWORD(x), DWORD(x + 4)))
#else /* ALIGNMENT_WORKAROUND || BIGENDIAN */
#define WORD(x) (u16)(*(const u16 *)(x))
#define DWORD(x) (u32)(*(const u32 *)(x))
#define QWORD(x) (*(const u64 *)(x))
#endif /* ALIGNMENT_WORKAROUND || BIGENDIAN */

		static const char *bad_index = "<BAD INDEX>";

		class dmi_error : public std::exception
		{
		public:
			dmi_error(std::string str) : str_{ str }
			{
			}

			inline std::string erro_str() const
			{
				return str_;
			}
		private:
			std::string str_;
		};


		static int myread(int fd, u8 *buf, size_t count, const char *prefix)
		{
			ssize_t r = 1;
			size_t r2 = 0;

			while (r2 != count && r != 0)
			{
				r = read(fd, buf + r2, count - r2);
				if (r == -1)
				{
					if (errno != EINTR)
					{
						perror(prefix);
						return -1;
					}
				}
				else
					r2 += r;
			}

			if (r2 != count)
			{
				fprintf(stderr, "%s: Unexpected end of file\n", prefix);
				return -1;
			}

			return 0;
		}

		/*
		* Reads all of file from given offset, up to max_len bytes.
		* A buffer of at most max_len bytes is allocated by this function, and
		* needs to be freed by the caller.
		* This provides a similar usage model to mem_chunk()
		*
		* Returns a pointer to the allocated buffer, or NULL on error, and
		* sets max_len to the length actually read.
		*/
		static void *read_file(off_t base, size_t *max_len, const char *filename)
		{
			struct stat statbuf;
			int fd;
			u8 *p;

			/*
			* Don't print error message on missing file, as we will try to read
			* files that may or may not be present.
			*/
			if ((fd = open(filename, O_RDONLY)) == -1)
			{
				if (errno != ENOENT)
					perror(filename);
				return NULL;
			}

			/*
			* Check file size, don't allocate more than can be read.
			*/
			if (fstat(fd, &statbuf) == 0)
			{
				if (base >= statbuf.st_size)
				{
					fprintf(stderr, "%s: Can't read data beyond EOF\n",
						filename);
					p = NULL;
					goto out;
				}
				if (*max_len > (size_t)statbuf.st_size - base)
					*max_len = statbuf.st_size - base;
			}

			if ((p = (u8 *)malloc(*max_len)) == NULL)
			{
				perror("malloc");
				goto out;
			}

			if (lseek(fd, base, SEEK_SET) == -1)
			{
				fprintf(stderr, "%s: ", filename);
				perror("lseek");
				goto err_free;
			}

			if (myread(fd, p, *max_len, filename) == 0)
				goto out;

		err_free:
			free(p);
			p = NULL;

		out:
			if (close(fd) == -1)
				perror(filename);

			return p;
		}

		inline int checksum(const u8 *buf, size_t len)
		{
			u8 sum = 0;
			size_t a;

			for (a = 0; a < len; a++)
				sum += buf[a];
			return (sum == 0);
		}


		struct dmi_header
		{
			u8 type;
			u8 length;
			u16 handle;
			u8 *data;
		};

		static const char *dmi_string(const struct dmi_header *dm, u8 s)
		{
			char *bp = (char *)dm->data;
			size_t i, len;

			if (s == 0)
				return "Not Specified";

			bp += dm->length;
			while (s > 1 && *bp)
			{
				bp += strlen(bp);
				bp++;
				s--;
			}

			if (!*bp)
				return bad_index;

			/* ASCII filtering */
			len = strlen(bp);
			for (i = 0; i < len; i++)
				if (bp[i] < 32 || bp[i] == 127)
					bp[i] = '.';

			return bp;
		}

		static std::string dmi_system_uuid(const u8 *p, u16 ver)
		{
			std::string uuid;
			int only0xFF = 1, only0x00 = 1;
			int i;

			for (i = 0; i < 16 && (only0x00 || only0xFF); i++)
			{
				if (p[i] != 0x00) only0x00 = 0;
				if (p[i] != 0xFF) only0xFF = 0;
			}

			if (only0xFF)
			{
				return uuid;
			}
			if (only0x00)
			{
				return uuid;
			}

			/*
			* As of version 2.6 of the SMBIOS specification, the first 3
			* fields of the UUID are supposed to be encoded on little-endian.
			* The specification says that this is the defacto standard,
			* however I've seen systems following RFC 4122 instead and use
			* network byte order, so I am reluctant to apply the byte-swapping
			* for older versions.
			*/

			char temp[48];
			if (ver >= 0x0206)
				sprintf(temp, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
					p[3], p[2], p[1], p[0], p[5], p[4], p[7], p[6],
					p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
			else
				sprintf(temp, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
					p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
					p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);

			uuid = temp;

			return uuid;
		}

		static bool dmi_decode(const struct dmi_header *h, u16 ver, std::unordered_map<std::string, std::string> &map)
		{
			const u8 *data = h->data;

			/*
			* Note: DMI types 37 and 42 are untested
			*/
			switch (h->type)
			{
			case 1: /* System Information */
				if (h->length < 0x08)
					return false;
				map["Manufacturer"] = dmi_string(h, data[0x04]);
				map["Product Name"] = dmi_string(h, data[0x05]);
				map["Version"] = dmi_string(h, data[0x06]);
				map["Serial Number"] = dmi_string(h, data[0x07]);
				if (h->length < 0x19)
					return false;
				map["UUID"] = dmi_system_uuid(data + 0x08, ver);
				break;

			case 2: /* Base Board Information */
				if (h->length < 0x08)
					return false;
				map["Manufacturer"] = dmi_string(h, data[0x04]);
				map["Product Name"] = dmi_string(h, data[0x05]);
				map["Version"] = dmi_string(h, data[0x06]);
				map["Serial Number"] = dmi_string(h, data[0x07]);
				break;

			default:
				return false;
			}
			return true;
		}

		static void to_dmi_header(struct dmi_header *h, u8 *data)
		{
			h->type = data[0];
			h->length = data[1];
			h->handle = WORD(data + 2);
			h->data = data;
		}

		enum class DecodeType
		{
			SYSTEM = 1,
			BASEBOARD = 2
		};

		static bool dmi_table_decode(u8 *buf, u32 len, u16 num, u16 ver, u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map)
		{
			u8 *data;
			int i = 0;

			data = buf;
			while ((i < num || !num)
				&& data + 4 <= buf + len) /* 4 is the length of an SMBIOS structure header */
			{
				u8 *next;
				struct dmi_header h;
				int display;

				to_dmi_header(&h, data);

				/*
				* If a short entry is found (less than 4 bytes), not only it
				* is invalid, but we cannot reliably locate the next entry.
				* Better stop at this point, and let the user know his/her
				* table is broken.
				*/
				if (h.length < 4)
				{
					return false;
				}
				i++;


				/* Look for the next handle */
				next = data + h.length;
				while ((unsigned long)(next - buf + 1) < len
					&& (next[0] != 0 || next[1] != 0))
					next++;
				next += 2;

				/* Make sure the whole structure fits in the table */
				if ((unsigned long)(next - buf) > len)
				{
					data = next;
					return false;
				}

				if ((u8)type == h.type)
					return dmi_decode(&h, ver, map);

				data = next;

				/* SMBIOS v3 requires stopping at this marker */
				if (h.type == 127 && (flags & FLAG_STOP_AT_EOT))
					return false;
			}
			return false;
		}

		static bool dmi_table(off_t base, u32 len, u16 num, u32 ver, const char *devmem,
			u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map)
		{
			u8 *buf;

			if (ver > SUPPORTED_SMBIOS_VER)
			{
				printf("# SMBIOS implementations newer than version %u.%u.%u are not\n"
					"# fully supported by this version of dmidecode.\n",
					SUPPORTED_SMBIOS_VER >> 16,
					(SUPPORTED_SMBIOS_VER >> 8) & 0xFF,
					SUPPORTED_SMBIOS_VER & 0xFF);
			}

			if (flags & FLAG_NO_FILE_OFFSET)
			{
				/*
				* When reading from sysfs or from a dump file, the file may be
				* shorter than announced. For SMBIOS v3 this is expcted, as we
				* only know the maximum table size, not the actual table size.
				* For older implementations (and for SMBIOS v3 too), this
				* would be the result of the kernel truncating the table on
				* parse error.
				*/
				size_t size = len;
				buf = (u8 *)read_file(flags & FLAG_NO_FILE_OFFSET ? 0 : base,
					&size, devmem);
				if (num && size != (size_t)len)
				{
					return false;
				}
				len = size;
			}
			else
				return false;

			if (buf == NULL)
			{
				return false;
			}
			bool ret = dmi_table_decode(buf, len, num, ver >> 8, flags, type, map);

			free(buf);
			return ret;
		}

		class DmiBaseSmBios
		{
		public:
			virtual ~DmiBaseSmBios() {}
			virtual bool smbios_decode(u8 *buf, const char *devmem, u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map) = 0;
		};

		class DmiSmBios3 : public DmiBaseSmBios
		{
		public:
			virtual bool smbios_decode(u8 *buf, const char *devmem, u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map)
			{
				u32 ver;
				u64 offset;

				/* Don't let checksum run beyond the buffer */
				if (buf[0x06] > 0x20)
				{
					return false;
				}

				if (!checksum(buf, buf[0x06]))
					return false;

				ver = (buf[0x07] << 16) + (buf[0x08] << 8) + buf[0x09];

				offset = QWORD(buf + 0x10);
				if (!(flags & FLAG_NO_FILE_OFFSET) && offset.h && sizeof(off_t) < 8)
				{
					return false;
				}

				return dmi_table(((off_t)offset.h << 32) | offset.l,
					DWORD(buf + 0x0C), 0, ver, devmem, flags | FLAG_STOP_AT_EOT, type, map);
			}
		};

		class DmiSmBios : public DmiBaseSmBios
		{
		public:
			virtual bool smbios_decode(u8 *buf, const char *devmem, u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map)
			{
				u16 ver;

				/* Don't let checksum run beyond the buffer */
				if (buf[0x05] > 0x20)
				{
					return false;
				}

				if (!checksum(buf, buf[0x05])
					|| memcmp(buf + 0x10, "_DMI_", 5) != 0
					|| !checksum(buf + 0x10, 0x0F))
					return false;

				ver = (buf[0x06] << 8) + buf[0x07];
				/* Some BIOS report weird SMBIOS version, fix that up */
				switch (ver)
				{
				case 0x021F:
				case 0x0221:
					ver = 0x0203;
					break;
				case 0x0233:
					ver = 0x0206;
					break;
				}

				dmi_table(DWORD(buf + 0x18), WORD(buf + 0x16), WORD(buf + 0x1C),
					ver << 8, devmem, flags, type, map);

				return true;
			}
		};

		class DmiLegacyBios : public DmiBaseSmBios
		{
		public:
			virtual bool smbios_decode(u8 *buf, const char *devmem, u32 flags, enum DecodeType type, std::unordered_map<std::string, std::string> &map)
			{
				if (!checksum(buf, 0x0F))
					return false;

				dmi_table(DWORD(buf + 0x08), WORD(buf + 0x06), WORD(buf + 0x0C),
					((buf[0x0E] & 0xF0) << 12) + ((buf[0x0E] & 0x0F) << 8),
					devmem, flags, type, map);

				return true;
			}
		};

		class DmiBaseDecoder
		{
		public:
			virtual ~DmiBaseDecoder()
			{
				if (buf != nullptr)
					free(buf);
			}

			virtual std::unordered_map<std::string, std::string> decode(enum DecodeType type) = 0;

			u8 * buf;
			const char *devmem;
			u32 flags;
			DmiBaseSmBios *bios;
			std::unordered_map<std::string, std::string> map;
		};

		class DmiSysDecoder final : public DmiBaseDecoder, public singleton<DmiSysDecoder>
		{
		public:
			DmiSysDecoder() : size(0x20)
			{
				buf = (u8 *)read_file(0, &size, SYS_ENTRY_FILE);
				if (buf != nullptr)
				{
					devmem = SYS_TABLE_FILE;
					flags |= FLAG_NO_FILE_OFFSET;
				}
				else
				{
					throw dmi_error("DmiSysDecoder read_file error");
				}

				if (size >= 24 && memcmp(buf, "_SM3_", 5) == 0)
				{
					bios = new DmiSmBios3;
				}
				else if (size >= 31 && memcmp(buf, "_SM_", 4) == 0)
				{
					bios = new DmiSmBios;
				}
				else if (size >= 15 && memcmp(buf, "_DMI_", 5) == 0)
				{
					bios = new DmiLegacyBios;
				}
			}

			~DmiSysDecoder()
			{
				free(bios);
			}

			virtual std::unordered_map<std::string, std::string> decode(enum DecodeType type)
			{
				if (!bios->smbios_decode(buf, devmem, flags, type, map))
					throw dmi_error("smbios_decode error");
				return map;
			}

			size_t size;
		};
	}
}
#endif

#endif