set(SOURCES
	PoolURI.cpp PoolURI.h
	PoolClient.h
	PoolManager.h PoolManager.cpp
	testing/SimulateClient.h testing/SimulateClient.cpp
	stratum/EthStratumClient.h stratum/EthStratumClient.cpp
	getwork/EthGetworkClient.h getwork/EthGetworkClient.cpp
)

hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

add_library(poolprotocols ${SOURCES})
target_link_libraries(poolprotocols PRIVATE devcore ethminer-buildinfo ethash::ethash Boost::system jsoncpp_static OpenSSL::SSL OpenSSL::Crypto)
if(WIN32)
	# Needed for OpenSSL and Boost.Asio socket APIs on Windows/MinGW.
	target_link_libraries(poolprotocols PUBLIC ws2_32 mswsock)
endif()
target_include_directories(poolprotocols PRIVATE ..)
