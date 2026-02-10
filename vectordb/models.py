from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Column
import json
from sqlalchemy import TEXT


class User(SQLModel, table=True):
    """User model stored in the database."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, unique=True)  # Supabase user ID
    email: Optional[str] = Field(default=None, index=True)  # User's email address
    first_name: Optional[str] = Field(default=None)  # User's first name
    last_name: Optional[str] = Field(default=None)  # User's last name
    admin: bool = Field(default=False)  # Whether user is an admin
    teacher: bool = Field(default=False)  # Whether user is a teacher/instructor
    icon_shape: str = Field(default="circle")  # Profile icon shape: circle, square, or hex
    icon_color: str = Field(default="#4f46e5")  # Profile icon color (hex code)
    initials: Optional[str] = Field(default=None, max_length=2)  # User's custom initials (max 2 chars)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Question(SQLModel, table=True):
    """Question model stored in the database."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(default="")  # Question title (e.g., Invert a Linked List)
    text: str = Field(index=True)
    tags: str = Field(default="")  # Question tags (e.g., recursion, sorting, runtime analysis)
    keywords: str = Field(default="")  # Stored as comma-separated string
    school: str = Field(default="")  # School name (e.g., UCSB)
    course: str = Field(default="")  # Course name (kept for backward compatibility)
    course_type: str = Field(default="")  # Course type (e.g., intro CS, intermediate CS, linear algebra)
    question_type: str = Field(default="")  # Question type (e.g., mcq, fr, short answer)
    blooms_taxonomy: str = Field(default="")  # Bloom's taxonomy level (e.g., Remembering, Understanding)
    answer_choices: str = Field(sa_column=Column(TEXT), default="[]")  # JSON array of answer choices
    correct_answer: str = Field(default="")  # The correct answer text
    pdf_url: Optional[str] = Field(default=None)  # URL to PDF in Supabase bucket
    source_pdf: Optional[str] = Field(default=None)  # Original PDF filename
    image_url: Optional[str] = Field(default=None)  # URL to image stored in Supabase bucket
    user_id: str = Field(index=True)  # Supabase user ID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_verified: bool = Field(default=False)  # Whether question in database is verified


class CourseStudent(SQLModel, table=True):
    """Association table for many-to-many relationship between courses and students."""
    __tablename__ = "course_student"
    
    course_id: Optional[int] = Field(default=None, foreign_key="course.id", primary_key=True)
    student_id: str = Field(foreign_key="user.user_id", primary_key=True)  # Supabase user ID
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)


class Assignment(SQLModel, table=True):
    """Assignment model for course assignments."""
    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: Optional[str] = Field(default=None)  # Foreign key to Course Tree Node (null for now)
    instructor_email: str = Field(default="")  # Email of instructor who created assignment
    instructor_id: str = Field(default="")  # ID of instructor user who created assignment
    course: str = Field(default="")  # ID/name of course the assignment was created in
    course_id: int = Field(foreign_key="course.id", index=True)
    title: str = Field(index=True)
    type: str = Field(default="Other")  # Homework, Quiz, Lab, Exam, Reading, Other
    description: str = Field(default="")
    release_date: Optional[datetime] = Field(default=None)  # Visibility trigger for student portal
    due_date_soft: Optional[datetime] = Field(default=None)  # Target date; no points deducted
    due_date_hard: Optional[datetime] = Field(default=None)  # Final cut-off for Autograder
    late_policy_id: Optional[str] = Field(default=None)  # Reference to policy template
    assignment_questions: str = Field(sa_column=Column(TEXT), default="[]")  # JSON array of question IDs
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Course(SQLModel, table=True):
    """Course model stored in the database."""
    id: Optional[int] = Field(default=None, primary_key=True)
    course_name: str = Field(index=True)
    school_name: str = Field(default="")
    instructor_id: str = Field(foreign_key="user.user_id", index=True)  # Supabase user ID of instructor
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
