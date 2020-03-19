#include "../../include/Primitives/allocator.hpp"
#include "../../include/Longinus/longinus_c.h"
#ifdef PYTHON_SUPPORT
#include <Python.h>

using namespace glasssix::longinus;

typedef struct {
	PyObject_HEAD
	//list<void *> *alloc_mems;
	int status;
	glasssix::longinus::LonginusDetector* handle;
	//PyObject* _CUBLAS_OP;
	//PyObject* _CUBLAS_FILL_MODE;
	//PyObject* _CUBLAS_DIAG;
	//PyObject* _CUBLAS_SIDE_MODE;
	int validHandle;
} pyLonginusDetector;

static void ReleaseInstance(PyObject *obj) 
{
	auto old_ptr = static_cast<pyLonginusDetector*>(PyCapsule_GetPointer(obj, "pyLonginus"));
	Longinus_ReleaseInstance(old_ptr->handle);
	glasssix::memory::heap_free(old_ptr);
	printf("Destruction done!\n");
}

static PyObject* NewInstance(PyObject* self, PyObject *args/*, PyObject* kwds*/)
{
	int device;
	if (!PyArg_ParseTuple(args, "i", &device))
		return nullptr;
	printf("Works on GPU: %i\n", device);
	auto ptr = static_cast<pyLonginusDetector*>(glasssix::memory::heap_alloc(sizeof(pyLonginusDetector)));
	ptr->handle = Longinus_NewInstance(device);
	printf("Init done!\n");
	return PyCapsule_New(ptr, "pyLonginus", 1 ? ReleaseInstance : nullptr);
}

static PyObject* GetVersion(PyObject *self, PyObject *args)
{
	const char* ver = Longinus_getVersion();
	printf("%s", ver);
	return Py_BuildValue("s", ver);
}


static PyMethodDef pyLonginus_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{ "new_instance", (PyCFunction)NewInstance, METH_VARARGS, nullptr },
	{ "get_version", (PyCFunction)GetVersion, METH_NOARGS, nullptr },
	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pyLonginus_module = {
	PyModuleDef_HEAD_INIT,
	"pyLonginus",                        // Module name to use with Python import statements
	"Python wrapper for Longinus",  // Module description
	0,
	pyLonginus_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pyLonginus() {
	return PyModule_Create(&pyLonginus_module);
}


#endif // !PYTHON_SUPPORT


