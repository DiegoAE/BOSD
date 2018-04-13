#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Labels
#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <ForwardBackward.hpp>
#include <exception>

BOOST_AUTO_TEST_CASE( single_segment ) {
    Labels segments;
    BOOST_REQUIRE_THROW(segments.setLabel(0, 2), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(100, 2, -2), std::logic_error);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(100, 101));
}

BOOST_AUTO_TEST_CASE( duplicate_segment ) {
    Labels segments;
    segments.setLabel(50, 10);
    BOOST_REQUIRE_THROW(segments.setLabel(50, 5), std::logic_error);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(100, 10));
    BOOST_REQUIRE_THROW(segments.setLabel(100, 20), std::logic_error);
}

BOOST_AUTO_TEST_CASE( nonoverlapping_segments ) {
    Labels segments;
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 10));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(99, 50));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(49, 40));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(199, 100));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(210, 6));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(201, 2));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(204, 3));
    BOOST_REQUIRE_THROW(segments.setLabel(80, 2), std::logic_error);
}

BOOST_AUTO_TEST_CASE( overlapping_segments ) {
    Labels segments;
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 5));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(20, 5));
    BOOST_REQUIRE_THROW(segments.setLabel(5, 2), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(8, 2), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(15, 15), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(15, 8), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(15, 7), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(22, 10), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(22, 20), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(30, 30), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(30, 11), std::logic_error);
    BOOST_REQUIRE_THROW(segments.setLabel(100, 100), std::logic_error);
}

BOOST_AUTO_TEST_CASE( consistency ) {
    Labels segments;
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 5, -1));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(20, 5, 5));
    BOOST_CHECK(segments.isConsistent(15, 5, 2));
    BOOST_CHECK(segments.isConsistent(9, 5, 2));
    BOOST_CHECK(!segments.isConsistent(20, 5, 2));
    BOOST_CHECK(segments.isConsistent(20, 5, 5));
    BOOST_CHECK(!segments.isConsistent(25, 15, 2));
    BOOST_CHECK(segments.isConsistent(15, 6, 2));
    BOOST_CHECK(segments.isConsistent(30, 10, 2));
}
