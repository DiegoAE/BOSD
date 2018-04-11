#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Labels
#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <HSMM.hpp>
#include <exception>

BOOST_AUTO_TEST_CASE( single_segment ) {
    hsmm::Labels segments;
    BOOST_REQUIRE_THROW(segments.setLabel(0, 2), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(100, 2, -2), std::logic_error);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(100, 101));
}

BOOST_AUTO_TEST_CASE( duplicate_segment ) {
    hsmm::Labels segments;
    segments.setLabel(50, 10);
    BOOST_REQUIRE_THROW(segments.setLabel(50, 5), std::logic_error);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(100, 10));
    BOOST_REQUIRE_THROW(segments.setLabel(100, 20), std::logic_error);
}

