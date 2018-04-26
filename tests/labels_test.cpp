#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Labels
#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <ForwardBackward.hpp>
#include <exception>
#include <set>

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

BOOST_AUTO_TEST_CASE( isLabel ) {
    Labels segments;
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 5));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(20, 5, 5));
    BOOST_CHECK(segments.isLabel(9, 5));
    BOOST_CHECK(!segments.isLabel(9, 5, 2));
    BOOST_CHECK(segments.isLabel(20, 5, 5));
    BOOST_CHECK(!segments.isLabel(20, 5, -1));
}

BOOST_AUTO_TEST_CASE( transition ) {
    Labels segments;
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 10, 1));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(99, 50, 3));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(49, 40, 2));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(199, 100));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(210, 6, 6));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(201, 2, 4));
    BOOST_REQUIRE_NO_THROW(segments.setLabel(204, 3, 5));
    BOOST_CHECK(segments.transition(1 , 2, 9));
    BOOST_CHECK(segments.transition(2 , 3, 49));
    BOOST_CHECK(!segments.transition(2 , 1, 49));
    BOOST_CHECK(segments.transition(4 , 5, 201));
    BOOST_CHECK(segments.transition(5 , 6, 204));
    BOOST_CHECK(!segments.transition(6 , 7, 210));
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 6; j++) {
            for(int t = 99; t <= 200; t++)
                BOOST_CHECK(!segments.transition(i , j, t));
            BOOST_CHECK(!segments.transition(i, j, 0));
            BOOST_CHECK(!segments.transition(i, j, 210));
        }
    std::set<int> transition_times = {9, 49, 99, 199, 201, 204};
    for(int i = 0; i <= 220; i++)
        if (transition_times.find(i) != transition_times.end())
            BOOST_CHECK(segments.transition(i));
        else
            BOOST_CHECK(!segments.transition(i));
}

BOOST_AUTO_TEST_CASE( firstSegment ) {
    Labels segments;
    BOOST_REQUIRE_THROW(segments.getFirstSegment(), std::logic_error);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(20, 5, 5));
    const ObservedSegment& first = segments.getFirstSegment();
    BOOST_CHECK(first.getEndingTime() == 20 && first.getHiddenState() == 5 &&
            first.getDuration() == 5);
    BOOST_REQUIRE_NO_THROW(segments.setLabel(9, 5));
    const ObservedSegment& first_ = segments.getFirstSegment();
    BOOST_CHECK(first_.getEndingTime() == 9 && first_.getHiddenState() == -1 &&
            first.getDuration() == 5);
}
