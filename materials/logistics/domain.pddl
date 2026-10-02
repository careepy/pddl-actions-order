(define (domain driver_shift_compliant_route_planning)
  (:requirements :strips :typing :negative-preconditions)
  (:types operational_resource_pool - object network_entity_group - object network_element - object candidate_root - object candidate_entity - candidate_root vehicle_unit - operational_resource_pool route_task_profile - operational_resource_pool operator_resource - operational_resource_pool equipment_resource - operational_resource_pool service_window - operational_resource_pool dispatch_slot - operational_resource_pool skill_certificate - operational_resource_pool compliance_certificate - operational_resource_pool cargo_payload - network_entity_group loading_container - network_entity_group operational_constraint - network_entity_group origin_node - network_element destination_node - network_element transport_leg - network_element driver_availability_state - candidate_entity dispatch_proposal - candidate_entity origin_candidate_availability - driver_availability_state destination_candidate_availability - driver_availability_state schedule_candidate - dispatch_proposal)
  (:predicates
    (candidate_created ?candidate - candidate_entity)
    (candidate_prepared ?candidate - candidate_entity)
    (vehicle_allocated ?candidate - candidate_entity)
    (dispatched ?candidate - candidate_entity)
    (dispatch_ready ?candidate - candidate_entity)
    (dispatch_allocated ?candidate - candidate_entity)
    (vehicle_available ?vehicle_unit - vehicle_unit)
    (candidate_vehicle_linked ?candidate - candidate_entity ?vehicle_unit - vehicle_unit)
    (route_task_profile_available ?route_task_profile - route_task_profile)
    (candidate_task_profile_linked ?candidate - candidate_entity ?route_task_profile - route_task_profile)
    (operator_available ?operator_resource - operator_resource)
    (candidate_operator_linked ?candidate - candidate_entity ?operator_resource - operator_resource)
    (cargo_available ?payload_unit - cargo_payload)
    (origin_payload_linked ?origin_availability - origin_candidate_availability ?payload_unit - cargo_payload)
    (destination_payload_linked ?destination_availability - destination_candidate_availability ?payload_unit - cargo_payload)
    (origin_availability_node_linked ?origin_availability - origin_candidate_availability ?origin_node - origin_node)
    (origin_node_ready ?origin_node - origin_node)
    (origin_node_occupied ?origin_node - origin_node)
    (origin_availability_confirmed ?origin_availability - origin_candidate_availability)
    (destination_availability_node_linked ?destination_availability - destination_candidate_availability ?destination_node - destination_node)
    (destination_node_ready ?destination_node - destination_node)
    (destination_node_occupied ?destination_node - destination_node)
    (destination_availability_confirmed ?destination_availability - destination_candidate_availability)
    (transport_leg_available ?transport_leg - transport_leg)
    (transport_leg_assembled ?transport_leg - transport_leg)
    (transport_leg_origin_linked ?transport_leg - transport_leg ?origin_node - origin_node)
    (transport_leg_destination_linked ?transport_leg - transport_leg ?destination_node - destination_node)
    (origin_leg_loaded ?transport_leg - transport_leg)
    (destination_leg_loaded ?transport_leg - transport_leg)
    (transport_leg_loading_authorized ?transport_leg - transport_leg)
    (candidate_origin_availability_linked ?candidate - schedule_candidate ?origin_availability - origin_candidate_availability)
    (candidate_destination_availability_linked ?candidate - schedule_candidate ?destination_availability - destination_candidate_availability)
    (candidate_transport_leg_linked ?candidate - schedule_candidate ?transport_leg - transport_leg)
    (loading_container_available ?loading_container - loading_container)
    (candidate_loading_container_linked ?candidate - schedule_candidate ?loading_container - loading_container)
    (loading_container_loaded ?loading_container - loading_container)
    (loading_container_leg_linked ?loading_container - loading_container ?transport_leg - transport_leg)
    (candidate_staged_for_loading ?candidate - schedule_candidate)
    (candidate_loading_verified ?candidate - schedule_candidate)
    (candidate_clearance_established ?candidate - schedule_candidate)
    (equipment_reserved ?candidate - schedule_candidate)
    (equipment_confirmed ?candidate - schedule_candidate)
    (route_sequence_context_set ?candidate - schedule_candidate)
    (route_sequence_validated ?candidate - schedule_candidate)
    (operational_constraint_available ?operational_constraint - operational_constraint)
    (candidate_operational_constraint_linked ?candidate - schedule_candidate ?operational_constraint - operational_constraint)
    (constraint_applied ?candidate - schedule_candidate)
    (operator_alignment_confirmed ?candidate - schedule_candidate)
    (compliance_validation_complete ?candidate - schedule_candidate)
    (equipment_available ?equipment_resource - equipment_resource)
    (candidate_equipment_linked ?candidate - schedule_candidate ?equipment_resource - equipment_resource)
    (service_window_available ?service_window - service_window)
    (candidate_service_window_linked ?candidate - schedule_candidate ?service_window - service_window)
    (skill_certificate_available ?skill_certificate - skill_certificate)
    (candidate_skill_certificate_linked ?candidate - schedule_candidate ?skill_certificate - skill_certificate)
    (compliance_certificate_available ?compliance_certificate - compliance_certificate)
    (candidate_compliance_certificate_linked ?candidate - schedule_candidate ?compliance_certificate - compliance_certificate)
    (dispatch_slot_available ?dispatch_slot - dispatch_slot)
    (candidate_dispatch_slot_linked ?candidate - candidate_entity ?dispatch_slot - dispatch_slot)
    (origin_availability_locked ?origin_availability - origin_candidate_availability)
    (destination_availability_locked ?destination_availability - destination_candidate_availability)
    (candidate_finalized ?candidate - schedule_candidate)
  )
  (:action create_candidate
    :parameters (?candidate - candidate_entity)
    :precondition
      (and
        (not
          (candidate_created ?candidate)
        )
        (not
          (dispatched ?candidate)
        )
      )
    :effect (candidate_created ?candidate)
  )
  (:action allocate_vehicle_unit
    :parameters (?candidate - candidate_entity ?vehicle_unit - vehicle_unit)
    :precondition
      (and
        (candidate_created ?candidate)
        (not
          (vehicle_allocated ?candidate)
        )
        (vehicle_available ?vehicle_unit)
      )
    :effect
      (and
        (vehicle_allocated ?candidate)
        (candidate_vehicle_linked ?candidate ?vehicle_unit)
        (not
          (vehicle_available ?vehicle_unit)
        )
      )
  )
  (:action attach_task_profile
    :parameters (?candidate - candidate_entity ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_created ?candidate)
        (vehicle_allocated ?candidate)
        (route_task_profile_available ?route_task_profile)
      )
    :effect
      (and
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (not
          (route_task_profile_available ?route_task_profile)
        )
      )
  )
  (:action confirm_candidate_preparedness
    :parameters (?candidate - candidate_entity ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_created ?candidate)
        (vehicle_allocated ?candidate)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (not
          (candidate_prepared ?candidate)
        )
      )
    :effect (candidate_prepared ?candidate)
  )
  (:action detach_task_profile
    :parameters (?candidate - candidate_entity ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_task_profile_linked ?candidate ?route_task_profile)
      )
    :effect
      (and
        (route_task_profile_available ?route_task_profile)
        (not
          (candidate_task_profile_linked ?candidate ?route_task_profile)
        )
      )
  )
  (:action assign_operator_resource
    :parameters (?candidate - candidate_entity ?operator_resource - operator_resource)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (operator_available ?operator_resource)
      )
    :effect
      (and
        (candidate_operator_linked ?candidate ?operator_resource)
        (not
          (operator_available ?operator_resource)
        )
      )
  )
  (:action release_operator_resource
    :parameters (?candidate - candidate_entity ?operator_resource - operator_resource)
    :precondition
      (and
        (candidate_operator_linked ?candidate ?operator_resource)
      )
    :effect
      (and
        (operator_available ?operator_resource)
        (not
          (candidate_operator_linked ?candidate ?operator_resource)
        )
      )
  )
  (:action assign_skill_certificate
    :parameters (?candidate - schedule_candidate ?skill_certificate - skill_certificate)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (skill_certificate_available ?skill_certificate)
      )
    :effect
      (and
        (candidate_skill_certificate_linked ?candidate ?skill_certificate)
        (not
          (skill_certificate_available ?skill_certificate)
        )
      )
  )
  (:action release_skill_certificate
    :parameters (?candidate - schedule_candidate ?skill_certificate - skill_certificate)
    :precondition
      (and
        (candidate_skill_certificate_linked ?candidate ?skill_certificate)
      )
    :effect
      (and
        (skill_certificate_available ?skill_certificate)
        (not
          (candidate_skill_certificate_linked ?candidate ?skill_certificate)
        )
      )
  )
  (:action assign_compliance_certificate
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (compliance_certificate_available ?compliance_certificate)
      )
    :effect
      (and
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (not
          (compliance_certificate_available ?compliance_certificate)
        )
      )
  )
  (:action release_compliance_certificate
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate)
    :precondition
      (and
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
      )
    :effect
      (and
        (compliance_certificate_available ?compliance_certificate)
        (not
          (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        )
      )
  )
  (:action activate_origin_node
    :parameters (?origin_availability - origin_candidate_availability ?origin_node - origin_node ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_prepared ?origin_availability)
        (candidate_task_profile_linked ?origin_availability ?route_task_profile)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (not
          (origin_node_ready ?origin_node)
        )
        (not
          (origin_node_occupied ?origin_node)
        )
      )
    :effect (origin_node_ready ?origin_node)
  )
  (:action lock_origin_availability
    :parameters (?origin_availability - origin_candidate_availability ?origin_node - origin_node ?operator_resource - operator_resource)
    :precondition
      (and
        (candidate_prepared ?origin_availability)
        (candidate_operator_linked ?origin_availability ?operator_resource)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (origin_node_ready ?origin_node)
        (not
          (origin_availability_locked ?origin_availability)
        )
      )
    :effect
      (and
        (origin_availability_locked ?origin_availability)
        (origin_availability_confirmed ?origin_availability)
      )
  )
  (:action load_origin_payload
    :parameters (?origin_availability - origin_candidate_availability ?origin_node - origin_node ?payload_unit - cargo_payload)
    :precondition
      (and
        (candidate_prepared ?origin_availability)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (cargo_available ?payload_unit)
        (not
          (origin_availability_locked ?origin_availability)
        )
      )
    :effect
      (and
        (origin_node_occupied ?origin_node)
        (origin_availability_locked ?origin_availability)
        (origin_payload_linked ?origin_availability ?payload_unit)
        (not
          (cargo_available ?payload_unit)
        )
      )
  )
  (:action unload_origin_payload
    :parameters (?origin_availability - origin_candidate_availability ?origin_node - origin_node ?route_task_profile - route_task_profile ?payload_unit - cargo_payload)
    :precondition
      (and
        (candidate_prepared ?origin_availability)
        (candidate_task_profile_linked ?origin_availability ?route_task_profile)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (origin_node_occupied ?origin_node)
        (origin_payload_linked ?origin_availability ?payload_unit)
        (not
          (origin_availability_confirmed ?origin_availability)
        )
      )
    :effect
      (and
        (origin_node_ready ?origin_node)
        (origin_availability_confirmed ?origin_availability)
        (cargo_available ?payload_unit)
        (not
          (origin_payload_linked ?origin_availability ?payload_unit)
        )
      )
  )
  (:action activate_destination_node
    :parameters (?destination_availability - destination_candidate_availability ?destination_node - destination_node ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_prepared ?destination_availability)
        (candidate_task_profile_linked ?destination_availability ?route_task_profile)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (not
          (destination_node_ready ?destination_node)
        )
        (not
          (destination_node_occupied ?destination_node)
        )
      )
    :effect (destination_node_ready ?destination_node)
  )
  (:action lock_destination_availability
    :parameters (?destination_availability - destination_candidate_availability ?destination_node - destination_node ?operator_resource - operator_resource)
    :precondition
      (and
        (candidate_prepared ?destination_availability)
        (candidate_operator_linked ?destination_availability ?operator_resource)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (destination_node_ready ?destination_node)
        (not
          (destination_availability_locked ?destination_availability)
        )
      )
    :effect
      (and
        (destination_availability_locked ?destination_availability)
        (destination_availability_confirmed ?destination_availability)
      )
  )
  (:action load_destination_payload
    :parameters (?destination_availability - destination_candidate_availability ?destination_node - destination_node ?payload_unit - cargo_payload)
    :precondition
      (and
        (candidate_prepared ?destination_availability)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (cargo_available ?payload_unit)
        (not
          (destination_availability_locked ?destination_availability)
        )
      )
    :effect
      (and
        (destination_node_occupied ?destination_node)
        (destination_availability_locked ?destination_availability)
        (destination_payload_linked ?destination_availability ?payload_unit)
        (not
          (cargo_available ?payload_unit)
        )
      )
  )
  (:action unload_destination_payload
    :parameters (?destination_availability - destination_candidate_availability ?destination_node - destination_node ?route_task_profile - route_task_profile ?payload_unit - cargo_payload)
    :precondition
      (and
        (candidate_prepared ?destination_availability)
        (candidate_task_profile_linked ?destination_availability ?route_task_profile)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (destination_node_occupied ?destination_node)
        (destination_payload_linked ?destination_availability ?payload_unit)
        (not
          (destination_availability_confirmed ?destination_availability)
        )
      )
    :effect
      (and
        (destination_node_ready ?destination_node)
        (destination_availability_confirmed ?destination_availability)
        (cargo_available ?payload_unit)
        (not
          (destination_payload_linked ?destination_availability ?payload_unit)
        )
      )
  )
  (:action assemble_transport_leg
    :parameters (?origin_availability - origin_candidate_availability ?destination_availability - destination_candidate_availability ?origin_node - origin_node ?destination_node - destination_node ?transport_leg - transport_leg)
    :precondition
      (and
        (origin_availability_locked ?origin_availability)
        (destination_availability_locked ?destination_availability)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (origin_node_ready ?origin_node)
        (destination_node_ready ?destination_node)
        (origin_availability_confirmed ?origin_availability)
        (destination_availability_confirmed ?destination_availability)
        (transport_leg_available ?transport_leg)
      )
    :effect
      (and
        (transport_leg_assembled ?transport_leg)
        (transport_leg_origin_linked ?transport_leg ?origin_node)
        (transport_leg_destination_linked ?transport_leg ?destination_node)
        (not
          (transport_leg_available ?transport_leg)
        )
      )
  )
  (:action assemble_origin_loaded_leg
    :parameters (?origin_availability - origin_candidate_availability ?destination_availability - destination_candidate_availability ?origin_node - origin_node ?destination_node - destination_node ?transport_leg - transport_leg)
    :precondition
      (and
        (origin_availability_locked ?origin_availability)
        (destination_availability_locked ?destination_availability)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (origin_node_occupied ?origin_node)
        (destination_node_ready ?destination_node)
        (not
          (origin_availability_confirmed ?origin_availability)
        )
        (destination_availability_confirmed ?destination_availability)
        (transport_leg_available ?transport_leg)
      )
    :effect
      (and
        (transport_leg_assembled ?transport_leg)
        (transport_leg_origin_linked ?transport_leg ?origin_node)
        (transport_leg_destination_linked ?transport_leg ?destination_node)
        (origin_leg_loaded ?transport_leg)
        (not
          (transport_leg_available ?transport_leg)
        )
      )
  )
  (:action assemble_destination_loaded_leg
    :parameters (?origin_availability - origin_candidate_availability ?destination_availability - destination_candidate_availability ?origin_node - origin_node ?destination_node - destination_node ?transport_leg - transport_leg)
    :precondition
      (and
        (origin_availability_locked ?origin_availability)
        (destination_availability_locked ?destination_availability)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (origin_node_ready ?origin_node)
        (destination_node_occupied ?destination_node)
        (origin_availability_confirmed ?origin_availability)
        (not
          (destination_availability_confirmed ?destination_availability)
        )
        (transport_leg_available ?transport_leg)
      )
    :effect
      (and
        (transport_leg_assembled ?transport_leg)
        (transport_leg_origin_linked ?transport_leg ?origin_node)
        (transport_leg_destination_linked ?transport_leg ?destination_node)
        (destination_leg_loaded ?transport_leg)
        (not
          (transport_leg_available ?transport_leg)
        )
      )
  )
  (:action assemble_fully_loaded_leg
    :parameters (?origin_availability - origin_candidate_availability ?destination_availability - destination_candidate_availability ?origin_node - origin_node ?destination_node - destination_node ?transport_leg - transport_leg)
    :precondition
      (and
        (origin_availability_locked ?origin_availability)
        (destination_availability_locked ?destination_availability)
        (origin_availability_node_linked ?origin_availability ?origin_node)
        (destination_availability_node_linked ?destination_availability ?destination_node)
        (origin_node_occupied ?origin_node)
        (destination_node_occupied ?destination_node)
        (not
          (origin_availability_confirmed ?origin_availability)
        )
        (not
          (destination_availability_confirmed ?destination_availability)
        )
        (transport_leg_available ?transport_leg)
      )
    :effect
      (and
        (transport_leg_assembled ?transport_leg)
        (transport_leg_origin_linked ?transport_leg ?origin_node)
        (transport_leg_destination_linked ?transport_leg ?destination_node)
        (origin_leg_loaded ?transport_leg)
        (destination_leg_loaded ?transport_leg)
        (not
          (transport_leg_available ?transport_leg)
        )
      )
  )
  (:action authorize_transport_leg_loading
    :parameters (?transport_leg - transport_leg ?origin_availability - origin_candidate_availability ?route_task_profile - route_task_profile)
    :precondition
      (and
        (transport_leg_assembled ?transport_leg)
        (origin_availability_locked ?origin_availability)
        (candidate_task_profile_linked ?origin_availability ?route_task_profile)
        (not
          (transport_leg_loading_authorized ?transport_leg)
        )
      )
    :effect (transport_leg_loading_authorized ?transport_leg)
  )
  (:action load_container_onto_leg
    :parameters (?candidate - schedule_candidate ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (candidate_transport_leg_linked ?candidate ?transport_leg)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_available ?loading_container)
        (transport_leg_assembled ?transport_leg)
        (transport_leg_loading_authorized ?transport_leg)
        (not
          (loading_container_loaded ?loading_container)
        )
      )
    :effect
      (and
        (loading_container_loaded ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (not
          (loading_container_available ?loading_container)
        )
      )
  )
  (:action stage_schedule_candidate
    :parameters (?candidate - schedule_candidate ?loading_container - loading_container ?transport_leg - transport_leg ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_loaded ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (not
          (origin_leg_loaded ?transport_leg)
        )
        (not
          (candidate_staged_for_loading ?candidate)
        )
      )
    :effect (candidate_staged_for_loading ?candidate)
  )
  (:action reserve_equipment_resource
    :parameters (?candidate - schedule_candidate ?equipment_resource - equipment_resource)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (equipment_available ?equipment_resource)
        (not
          (equipment_reserved ?candidate)
        )
      )
    :effect
      (and
        (equipment_reserved ?candidate)
        (candidate_equipment_linked ?candidate ?equipment_resource)
        (not
          (equipment_available ?equipment_resource)
        )
      )
  )
  (:action confirm_equipment_assignment
    :parameters (?candidate - schedule_candidate ?loading_container - loading_container ?transport_leg - transport_leg ?route_task_profile - route_task_profile ?equipment_resource - equipment_resource)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_loaded ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (origin_leg_loaded ?transport_leg)
        (equipment_reserved ?candidate)
        (candidate_equipment_linked ?candidate ?equipment_resource)
        (not
          (candidate_staged_for_loading ?candidate)
        )
      )
    :effect
      (and
        (candidate_staged_for_loading ?candidate)
        (equipment_confirmed ?candidate)
      )
  )
  (:action verify_loading_profile_nominal
    :parameters (?candidate - schedule_candidate ?skill_certificate - skill_certificate ?operator_resource - operator_resource ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_staged_for_loading ?candidate)
        (candidate_skill_certificate_linked ?candidate ?skill_certificate)
        (candidate_operator_linked ?candidate ?operator_resource)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (not
          (destination_leg_loaded ?transport_leg)
        )
        (not
          (candidate_loading_verified ?candidate)
        )
      )
    :effect (candidate_loading_verified ?candidate)
  )
  (:action verify_loading_profile_destination_flag
    :parameters (?candidate - schedule_candidate ?skill_certificate - skill_certificate ?operator_resource - operator_resource ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_staged_for_loading ?candidate)
        (candidate_skill_certificate_linked ?candidate ?skill_certificate)
        (candidate_operator_linked ?candidate ?operator_resource)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (destination_leg_loaded ?transport_leg)
        (not
          (candidate_loading_verified ?candidate)
        )
      )
    :effect (candidate_loading_verified ?candidate)
  )
  (:action establish_clearance_nominal
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_loading_verified ?candidate)
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (not
          (origin_leg_loaded ?transport_leg)
        )
        (not
          (destination_leg_loaded ?transport_leg)
        )
        (not
          (candidate_clearance_established ?candidate)
        )
      )
    :effect (candidate_clearance_established ?candidate)
  )
  (:action establish_clearance_origin_loaded
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_loading_verified ?candidate)
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (origin_leg_loaded ?transport_leg)
        (not
          (destination_leg_loaded ?transport_leg)
        )
        (not
          (candidate_clearance_established ?candidate)
        )
      )
    :effect
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_context_set ?candidate)
      )
  )
  (:action establish_clearance_destination_loaded
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_loading_verified ?candidate)
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (not
          (origin_leg_loaded ?transport_leg)
        )
        (destination_leg_loaded ?transport_leg)
        (not
          (candidate_clearance_established ?candidate)
        )
      )
    :effect
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_context_set ?candidate)
      )
  )
  (:action establish_clearance_full_load
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate ?loading_container - loading_container ?transport_leg - transport_leg)
    :precondition
      (and
        (candidate_loading_verified ?candidate)
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (candidate_loading_container_linked ?candidate ?loading_container)
        (loading_container_leg_linked ?loading_container ?transport_leg)
        (origin_leg_loaded ?transport_leg)
        (destination_leg_loaded ?transport_leg)
        (not
          (candidate_clearance_established ?candidate)
        )
      )
    :effect
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_context_set ?candidate)
      )
  )
  (:action finalize_candidate_direct
    :parameters (?candidate - schedule_candidate)
    :precondition
      (and
        (candidate_clearance_established ?candidate)
        (not
          (route_sequence_context_set ?candidate)
        )
        (not
          (candidate_finalized ?candidate)
        )
      )
    :effect
      (and
        (candidate_finalized ?candidate)
        (dispatch_ready ?candidate)
      )
  )
  (:action attach_service_window
    :parameters (?candidate - schedule_candidate ?service_window - service_window)
    :precondition
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_context_set ?candidate)
        (service_window_available ?service_window)
      )
    :effect
      (and
        (candidate_service_window_linked ?candidate ?service_window)
        (not
          (service_window_available ?service_window)
        )
      )
  )
  (:action validate_route_sequence
    :parameters (?candidate - schedule_candidate ?origin_availability - origin_candidate_availability ?destination_availability - destination_candidate_availability ?route_task_profile - route_task_profile ?service_window - service_window)
    :precondition
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_context_set ?candidate)
        (candidate_service_window_linked ?candidate ?service_window)
        (candidate_origin_availability_linked ?candidate ?origin_availability)
        (candidate_destination_availability_linked ?candidate ?destination_availability)
        (origin_availability_confirmed ?origin_availability)
        (destination_availability_confirmed ?destination_availability)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (not
          (route_sequence_validated ?candidate)
        )
      )
    :effect (route_sequence_validated ?candidate)
  )
  (:action finalize_candidate_after_sequence_validation
    :parameters (?candidate - schedule_candidate)
    :precondition
      (and
        (candidate_clearance_established ?candidate)
        (route_sequence_validated ?candidate)
        (not
          (candidate_finalized ?candidate)
        )
      )
    :effect
      (and
        (candidate_finalized ?candidate)
        (dispatch_ready ?candidate)
      )
  )
  (:action apply_operational_constraint
    :parameters (?candidate - schedule_candidate ?operational_constraint - operational_constraint ?route_task_profile - route_task_profile)
    :precondition
      (and
        (candidate_prepared ?candidate)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (operational_constraint_available ?operational_constraint)
        (candidate_operational_constraint_linked ?candidate ?operational_constraint)
        (not
          (constraint_applied ?candidate)
        )
      )
    :effect
      (and
        (constraint_applied ?candidate)
        (not
          (operational_constraint_available ?operational_constraint)
        )
      )
  )
  (:action confirm_operator_alignment
    :parameters (?candidate - schedule_candidate ?operator_resource - operator_resource)
    :precondition
      (and
        (constraint_applied ?candidate)
        (candidate_operator_linked ?candidate ?operator_resource)
        (not
          (operator_alignment_confirmed ?candidate)
        )
      )
    :effect (operator_alignment_confirmed ?candidate)
  )
  (:action verify_compliance_certificate
    :parameters (?candidate - schedule_candidate ?compliance_certificate - compliance_certificate)
    :precondition
      (and
        (operator_alignment_confirmed ?candidate)
        (candidate_compliance_certificate_linked ?candidate ?compliance_certificate)
        (not
          (compliance_validation_complete ?candidate)
        )
      )
    :effect (compliance_validation_complete ?candidate)
  )
  (:action finalize_candidate_after_compliance_validation
    :parameters (?candidate - schedule_candidate)
    :precondition
      (and
        (compliance_validation_complete ?candidate)
        (not
          (candidate_finalized ?candidate)
        )
      )
    :effect
      (and
        (candidate_finalized ?candidate)
        (dispatch_ready ?candidate)
      )
  )
  (:action mark_origin_availability_dispatch_ready
    :parameters (?origin_availability - origin_candidate_availability ?transport_leg - transport_leg)
    :precondition
      (and
        (origin_availability_locked ?origin_availability)
        (origin_availability_confirmed ?origin_availability)
        (transport_leg_assembled ?transport_leg)
        (transport_leg_loading_authorized ?transport_leg)
        (not
          (dispatch_ready ?origin_availability)
        )
      )
    :effect (dispatch_ready ?origin_availability)
  )
  (:action mark_destination_availability_dispatch_ready
    :parameters (?destination_availability - destination_candidate_availability ?transport_leg - transport_leg)
    :precondition
      (and
        (destination_availability_locked ?destination_availability)
        (destination_availability_confirmed ?destination_availability)
        (transport_leg_assembled ?transport_leg)
        (transport_leg_loading_authorized ?transport_leg)
        (not
          (dispatch_ready ?destination_availability)
        )
      )
    :effect (dispatch_ready ?destination_availability)
  )
  (:action allocate_dispatch_slot_for_candidate
    :parameters (?candidate - candidate_entity ?dispatch_slot - dispatch_slot ?route_task_profile - route_task_profile)
    :precondition
      (and
        (dispatch_ready ?candidate)
        (candidate_task_profile_linked ?candidate ?route_task_profile)
        (dispatch_slot_available ?dispatch_slot)
        (not
          (dispatch_allocated ?candidate)
        )
      )
    :effect
      (and
        (dispatch_allocated ?candidate)
        (candidate_dispatch_slot_linked ?candidate ?dispatch_slot)
        (not
          (dispatch_slot_available ?dispatch_slot)
        )
      )
  )
  (:action execute_origin_dispatch
    :parameters (?origin_availability - origin_candidate_availability ?vehicle_unit - vehicle_unit ?dispatch_slot - dispatch_slot)
    :precondition
      (and
        (dispatch_allocated ?origin_availability)
        (candidate_vehicle_linked ?origin_availability ?vehicle_unit)
        (candidate_dispatch_slot_linked ?origin_availability ?dispatch_slot)
        (not
          (dispatched ?origin_availability)
        )
      )
    :effect
      (and
        (dispatched ?origin_availability)
        (vehicle_available ?vehicle_unit)
        (dispatch_slot_available ?dispatch_slot)
      )
  )
  (:action execute_destination_dispatch
    :parameters (?destination_availability - destination_candidate_availability ?vehicle_unit - vehicle_unit ?dispatch_slot - dispatch_slot)
    :precondition
      (and
        (dispatch_allocated ?destination_availability)
        (candidate_vehicle_linked ?destination_availability ?vehicle_unit)
        (candidate_dispatch_slot_linked ?destination_availability ?dispatch_slot)
        (not
          (dispatched ?destination_availability)
        )
      )
    :effect
      (and
        (dispatched ?destination_availability)
        (vehicle_available ?vehicle_unit)
        (dispatch_slot_available ?dispatch_slot)
      )
  )
  (:action execute_schedule_dispatch
    :parameters (?candidate - schedule_candidate ?vehicle_unit - vehicle_unit ?dispatch_slot - dispatch_slot)
    :precondition
      (and
        (dispatch_allocated ?candidate)
        (candidate_vehicle_linked ?candidate ?vehicle_unit)
        (candidate_dispatch_slot_linked ?candidate ?dispatch_slot)
        (not
          (dispatched ?candidate)
        )
      )
    :effect
      (and
        (dispatched ?candidate)
        (vehicle_available ?vehicle_unit)
        (dispatch_slot_available ?dispatch_slot)
      )
  )
)
