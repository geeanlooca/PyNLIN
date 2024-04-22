from box import Box

from pynlin.fiber import Fiber, MMFiber

fiber_json = """
{
    "core_diameter": 1.4E-5,
    "ncl": 1.46,
    "nco": 1.4640994785399117,
    "delta": 0.0028000000000000004,
    "reference_lambda": 1.55E-6,
    "losses": [
        2.26786883e-06,
        -7.12461042e-03,
        5.78789219e+00
    ],
    "overlap_integrals": [
        [
            7.0132662979717083E+9,
            3.9457164610220423E+9
        ],
        [
            3.9457164610220394E+9,
            5.6725623567508116E+9
        ]
    ],
    "mode_names": [
        "LP(0,1)",
        "LP(1,1)"
    ],
    "modes": 2,
    "effective_areas": [
        [
            142.5869142155928,
            253.43939684428676
        ],
        [
            253.4393968442869,
            176.28717625464591
        ]
    ]
}
"""


fiber_config = Box.from_json(fiber_json)

mmf_sample = MMFiber(
    losses=fiber_config.losses,
    modes=fiber_config.modes,
    overlap_integrals=fiber_config.overlap_integrals,
)
